import os
import glob
import pydicom
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class CardiacSequenceDataset(Dataset):
    def __init__(self, root_dir, patient_ids, transform=None):
        self.root_dir = root_dir
        self.patient_ids = patient_ids
        self.transform = transform
        self.patient_sequences = []

        for p_id in self.patient_ids:
            patient_path = os.path.join(self.root_dir, p_id)
            if os.path.exists(patient_path):
                files = [
                    os.path.join(patient_path, f)
                    for f in os.listdir(patient_path)
                    if not f.startswith('.') and not f.endswith(('.md', '.txt', '.csv'))
                ]
                files.sort()
                if files:
                    self.patient_sequences.append(files)

    def __len__(self):
        return len(self.patient_sequences)

    def __getitem__(self, idx):
        sequence_paths = self.patient_sequences[idx]
        sequence_tensors = []

        for dcm_path in sequence_paths:
            dcm = pydicom.dcmread(dcm_path)
            image_array = dcm.pixel_array.astype(np.float32)

            # Min-max scaling
            denom = np.max(image_array) - np.min(image_array) + 1e-8
            image_array = (image_array - np.min(image_array)) / denom

            tensor_image = torch.tensor(image_array).unsqueeze(0)
            if self.transform:
                tensor_image = self.transform(tensor_image)

            sequence_tensors.append(tensor_image)

        return torch.stack(sequence_tensors)