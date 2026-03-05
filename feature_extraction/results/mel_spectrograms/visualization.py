import numpy as np
import matplotlib.pyplot as plt

mel = np.load("./train/2ea4_Counting1_mel.npy").squeeze(0)  # (128, T)

plt.imsave("mel.png", mel, cmap="magma", origin="lower")
