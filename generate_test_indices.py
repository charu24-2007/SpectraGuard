import h5py
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import os

HDF5_PATH = os.path.join("data", "raw", "RadioML2018", "GOLD_XYZ_OSC.0001_1024.hdf5")
NUM_CLASSES = 24
SNR_LEVELS = [6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
SAMPLES_PER_CLASS_PER_SNR = 1000 # 13 * 24 * 1000 = 312000

print("Loading Y and Z...")
with h5py.File(HDF5_PATH, "r") as f:
    Y_all = f["Y"][:].argmax(axis=1)
    Z_all = f["Z"][:, 0]

print("Selecting indices...")
indices_list = []
y_list = []
snr_list = []

for snr in SNR_LEVELS:
    snr_indices = np.where(Z_all == snr)[0]
    snr_labels = Y_all[snr_indices]

    for class_idx in range(NUM_CLASSES):
        sel = snr_indices[snr_labels == class_idx][:SAMPLES_PER_CLASS_PER_SNR]
        indices_list.append(sel)
        y_list.append(np.full(len(sel), class_idx, dtype=np.int64))
        snr_list.append(np.full(len(sel), snr, dtype=np.int64))

indices_pilot = np.concatenate(indices_list)
y_pilot = np.concatenate(y_list)
snr_pilot = np.concatenate(snr_list)

print("Splitting indices...")
# Split: 70% train, 15% val, 15% test
_, indices_temp, _, y_temp, _, snr_temp = train_test_split(
    indices_pilot, y_pilot, snr_pilot, test_size=0.30, random_state=42, stratify=y_pilot
)

_, indices_test, _, _, _, _ = train_test_split(
    indices_temp, y_temp, snr_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print(f"Got {len(indices_test)} test indices.")
np.save("backend/test_indices.npy", indices_test)
print("Saved to backend/test_indices.npy")
