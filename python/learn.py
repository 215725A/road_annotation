import argparse
import torch

from utils.yaml import load_config
from utils.setup import create_dir
from learning.dataset import RoadDataset

from torchvision import transforms
from torchvision import models
from torchvision.models import ResNet18_Weights
from torch import nn, optim

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

def main(yaml_path):
  config = load_config(yaml_path)
  img_path = config['image_path']
  csv_path = config['csv_path']
  epoch = config['epoch']
  model_path = config['output_model_path']

  create_dir(model_path)

  transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])

  dataset = RoadDataset(csv_path, img_path, transform)
  train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)
  train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=4)
  val_loader = DataLoader(val_data, batch_size=32, shuffle=False, num_workers=4)

  model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
  model.fc = nn.Linear(model.fc.in_features, 2)

  criterion = nn.CrossEntropyLoss()  # バイナリ分類なのでBCE
  optimizer = optim.Adam(model.parameters(), lr=0.001)

  # モデルの学習
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model.to(device)

  for e in range(epoch):  # エポック数を適宜調整
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
      inputs, labels = inputs.to(device), labels.to(device)

      optimizer.zero_grad()
      outputs = model(inputs)
      loss = criterion(outputs, labels)  # 予測値を.squeeze()でバッチサイズ次元を削除
      loss.backward()
      optimizer.step()

      running_loss += loss.item()

    print(f'Epoch [{e+1}/{epoch}], Loss: {running_loss / len(train_loader)}')

    # 検証
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
      for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        val_loss += loss.item()

    print(f'Validation Loss: {val_loss / len(val_loader)}')

    if (e + 1) % 5 == 0:
      output_model_path = model_path.format(epoch=e+1)
      torch.save(model.state_dict(), output_model_path)

  # モデルの保存
  output_model_path = model_path.format(epoch=epoch)
  torch.save(model.state_dict(), output_model_path)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-c', '--config', help='Select YML File')

  args = parser.parse_args()

  main(args.config)