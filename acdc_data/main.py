import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

path = "training/patient001/patient001_4d.nii.gz"
#path = "training/patient001/patient001_frame12_gt.nii.gz"

img = nib.load(path)
print(img.shape)
#(216, 256, 10, 30)

data = img.get_fdata()
print(data.size)

# Define a channel to look at
i = 8
#print(f"Plotting Layer {i} Channel {channel} of Image")

for channel in range(30):
    plt.imshow(data[:, :, i, channel], cmap='gray')
    plt.axis('off')
    plot_name = "plot_channel_" + str(channel) + "_i_" + str(i) + ".png"
    plt.savefig(plot_name)
