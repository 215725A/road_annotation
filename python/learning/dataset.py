import cv2

import pandas as pd

from torch.utils.data import Dataset

class RoadDataset(Dataset):
  def __init__(self, csv_file=None, img_dir=None, transform=None):
    if not csv_file is None:
      self.data = pd.read_csv(csv_file)
    
    if not img_dir is None:
      self.img_dir = img_dir
    self.transform = transform

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    img_name = self.data.iloc[idx, 0]
    image = cv2.imread(img_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    label = int(self.data.iloc[idx, 1])

    if self.transform:
        image = self.transform(image)

    return image, label