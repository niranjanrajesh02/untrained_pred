import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import Subset



def get_imagenet_dataloaders(train_dir, val_dir,batch_size=128, num_workers=4):
  train_tf = transforms.Compose([
      transforms.RandomResizedCrop(224),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485,0.456,0.406],
                          std=[0.229,0.224,0.225]),
  ])
  val_tf = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485,0.456,0.406],
                          std=[0.229,0.224,0.225]),
  ])
  
  trainsub_loader = DataLoader(
      datasets.ImageFolder(train_dir, train_tf),
      batch_size=batch_size, shuffle=False,
      num_workers=num_workers, pin_memory=True
  )


  # subset 1000 imgs for quick eval
  random_idx = torch.randperm(len(trainsub_loader.dataset))[:1000]

  train_loader = torch.utils.data.DataLoader(
      Subset(trainsub_loader.dataset, random_idx),
      batch_size=trainsub_loader.batch_size,
      shuffle=False,
      num_workers=trainsub_loader.num_workers,
  )

  val_loader = DataLoader(
      datasets.ImageFolder(val_dir, val_tf),
      batch_size=batch_size, shuffle=False,
      num_workers=num_workers, pin_memory=True
  )


  return train_loader, val_loader



def eval_model_val(model, val_dl, device_id=0):
  device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
  model.to(device)
  model.eval()

  correct_top1 = 0
  correct_top5 = 0
  total = 0

  with torch.no_grad():
    for images, labels in tqdm(val_dl, desc="Evaluating Model on Val Set"):
      images = images.to(device)
      labels = labels.to(device)

      logits = model(images)
      _, pred_top5 = logits.topk(5, 1, True, True)
      total += labels.size(0)
      correct = pred_top5.eq(labels.view(-1,1).expand_as(pred_top5))
      correct_top1 += correct[:, :1].sum().item()
      correct_top5 += correct.sum().item()

  val_acc_top1 = correct_top1 / total
  val_acc_top5 = correct_top5 / total

  return val_acc_top1, val_acc_top5





def eval_model_linear_probe(model, layer_name, train_dl, val_dl, device_id=0, num_epochs=1):
  device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
  model.to(device)
  layer = dict(model.named_modules())[layer_name]

  activations = []
  def hook_fn(module, input, output):
        activations.append(output.detach())

  handle = layer.register_forward_hook(hook_fn)

  for p in model.parameters():
    p.requires_grad = False


  # Classifier parameters
  num_dims = 4096
  num_labels = 1000
  lr = 0.01
  momentum = 0.9

  classifier = nn.Linear(num_dims, num_labels)
  classifier.to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(classifier.parameters(), lr=lr, momentum=momentum)
  scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
  

  # Early stopping parameters
  best_val_acc_top1 = 0.0
  best_val_acc_top5 = 0.0
  patience = 5
  patience_counter = 0

  classifier.train()
  model.eval()
  
  for epoch in tqdm(range(num_epochs), desc="Training Linear Probe"):
    epoch_accuracy = 0.0
    for images, labels in train_dl:
      images = images.to(device)
      labels = labels.to(device)

      with torch.no_grad():
        _ = model(images)
        del _
        
      acts = activations[0]
      activations = []

      # train classifier
      logits = classifier(acts)
      loss = criterion(logits, labels)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      correct = (logits.argmax(dim=1) == labels).sum().item()
      epoch_accuracy += correct / labels.size(0)

    epoch_accuracy /= len(train_dl)

    print(f'Epoch {epoch} completed. Training Accuracy: {epoch_accuracy}')
    scheduler.step()

    # after 5 epochs, start evaluating on validation set to track best model
    if epoch >= 5:
      # evaluate on validation set
      correct_top1 = 0
      correct_top5 = 0
      total = 0
      classifier.eval()
      with torch.no_grad():
        for images, labels in val_dl:
          images = images.to(device)
          labels = labels.to(device)

          _ = model(images)
          acts = activations[0].to(device)
          activations = []

          logits = classifier(acts)
          _, pred_top5 = logits.topk(5, 1, True, True)
          total += labels.size(0)
          correct = pred_top5.eq(labels.view(-1,1).expand_as(pred_top5))
          correct_top1 += correct[:, :1].sum().item()
          correct_top5 += correct.sum().item()

      val_acc_top1 = correct_top1 / total
      val_acc_top5 = correct_top5 / total
      # print(f'Epoch {epoch}: Validation Top-1 Accuracy: {val_acc_top1}, Top-5 Accuracy: {val_acc_top5}')
      if val_acc_top1 > best_val_acc_top1:
        best_val_acc_top1 = val_acc_top1
        best_val_acc_top5 = val_acc_top5
        patience_counter = 0
      else:
        patience_counter += 1
        if patience_counter >= patience:
          print(f"Early stopping at epoch {epoch}")
          break
      classifier.train()
  handle.remove()

  return classifier, (best_val_acc_top1, best_val_acc_top5) 