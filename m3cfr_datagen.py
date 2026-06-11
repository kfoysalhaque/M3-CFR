import os
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset, random_split


class M3CFRDataset(Dataset):
    def __init__(self, root_dir, env, subject, gestures, window_size=10, stride=5):
        self.root_dir = root_dir
        self.env = env
        self.subject = subject
        self.gestures = gestures
        self.window_size = window_size
        self.stride = stride

        self.label_map = {g: i for i, g in enumerate(gestures)}
        self.samples = []

        self._build_index()

    def _build_index(self):
        for gesture in self.gestures:
            folder = os.path.join(self.root_dir, self.env, self.subject, gesture)

            if not os.path.isdir(folder):
                print(f"Warning: missing folder {folder}")
                continue

            files = sorted([f for f in os.listdir(folder) if f.endswith(".mat")])

            for file in files:
                path = os.path.join(folder, file)

                mat = sio.loadmat(path)
                key = [k for k in mat.keys() if not k.startswith("__")][0]
                cfr = mat[key]

                num_frames = cfr.shape[-1]

                for start in range(0, num_frames - self.window_size + 1, self.stride):
                    self.samples.append((path, start, self.label_map[gesture]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, start, label = self.samples[idx]

        mat = sio.loadmat(path)
        key = [k for k in mat.keys() if not k.startswith("__")][0]
        cfr = mat[key]

        # Expected shape: 1024 x 8 x 8 x frames
        cfr_window = cfr[:, :, :, start:start + self.window_size]

        # 1024 x 8 x 8 x Tw -> Tw x 1024 x 8 x 8
        cfr_window = np.transpose(cfr_window, (3, 0, 1, 2))

        # Tw x 1024 x 8 x 8 -> Tw x 1024 x 64
        Tw, K, N, M = cfr_window.shape
        cfr_window = cfr_window.reshape(Tw, K, N * M)

        # Tw x 1024 x 64 -> Tw x 1024 x 128
        cfr_window = np.concatenate(
            [np.real(cfr_window), np.imag(cfr_window)],
            axis=-1
        )

        # Remove NaN and Inf values
        cfr_window = np.nan_to_num(
            cfr_window,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        # Per-sample normalization
        mean = np.mean(cfr_window)
        std = np.std(cfr_window)

        if std > 1e-8:
            cfr_window = (cfr_window - mean) / std
        else:
            cfr_window = cfr_window - mean

        cfr_window = cfr_window.astype(np.float32)

        x = torch.tensor(cfr_window, dtype=torch.float32)
        y = torch.tensor(label, dtype=torch.long)

        return x, y


def create_train_test_split(dataset, train_ratio=0.8, seed=42):
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size

    generator = torch.Generator().manual_seed(seed)

    train_dataset, test_dataset = random_split(
        dataset,
        [train_size, test_size],
        generator=generator
    )

    return train_dataset, test_dataset