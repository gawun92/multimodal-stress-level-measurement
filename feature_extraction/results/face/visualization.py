import numpy as np
import matplotlib.pyplot as plt

# select an npy file to visualize
data = np.load("./train/2ea4/Baseline_face.npy") 
fig, axes = plt.subplots(10, 10, figsize=(20, 20))  
for i, ax in enumerate(axes.flat):
    if i >= data.shape[0]:
        ax.axis('off')
        continue

    frame = data[i]
    if frame.max() == 0:  
        ax.axis('off')
        continue
    ax.scatter(frame[:, 0], frame[:, 1], s=0.3, c='steelblue')
    ax.invert_yaxis()
    ax.set_title(f"Frame {i}", fontsize=6)
    ax.axis('off')

plt.suptitle("All Frames - Face Landmarks", fontsize=14)
plt.tight_layout()
plt.show()