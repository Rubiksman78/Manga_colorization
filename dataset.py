import glob
import os
import random
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class ImageDataset(Dataset):
    def __init__(self, root, transform=None, unaligned=False, mode="train"):
        self.transform = transform
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, "A") + "\\*.*"))
        self.files_B = sorted(glob.glob(os.path.join(root, "B") + "\\*.*"))

    def __getitem__(self, index):
        try:
            item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]).convert('RGB'))

            if self.unaligned:
                item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]).convert('RGB'))
            else:
                item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]).convert('RGB'))   
            return {"A": item_A, "B": item_B}
        except:
            return None
        
    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
