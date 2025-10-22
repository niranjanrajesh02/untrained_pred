import pandas as pd
from my_utils.nsd_dataset import get_NSD_dataset

ds = get_NSD_dataset("./neural_data/tripleN/images")

print(f"Dataset length: {len(ds)}")

img = next(iter(ds))
print(f"Image shape: {img.shape}")
# display image
import matplotlib.pyplot as plt
plt.imshow(img.permute(1, 2, 0))
plt.axis('off')
