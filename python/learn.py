import argparse
import torch

from utils.yaml import load_config
from utils.setup import create_dir
from utils.plot import plot_loss, plot_accuracy, plot_precision, plot_recall, plot_f1_score
from learning.dataset import RoadDataset

from torchvision import transforms
from torchvision import models
from torchvision.models import ResNet18_Weights
from torch import nn, optim

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
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

  train_losses = []
  train_accuracies = []
  train_precisions = []
  train_recalls = []
  train_f1_scores = []

  val_losses = []
  val_accuracies = []
  val_precisions = []
  val_recalls = []
  val_f1_scores = []

  for e in range(epoch):  # エポック数を適宜調整
    model.train()
    running_loss = 0.0
    train_all_labels = []
    train_all_preds = []

    for inputs, labels in train_loader:
      inputs, labels = inputs.to(device), labels.to(device)

      optimizer.zero_grad()
      outputs = model(inputs)
      loss = criterion(outputs, labels)  # 予測値を.squeeze()でバッチサイズ次元を削除
      loss.backward()
      optimizer.step()

      running_loss += loss.item()

      _, preds = torch.max(outputs, 1)
      train_all_labels.extend(labels.cpu().numpy())
      train_all_preds.extend(preds.cpu().numpy())

    train_accuracy = (torch.tensor(train_all_preds) == torch.tensor(train_all_labels)).sum().item() / len(train_all_labels)
    train_precision = precision_score(train_all_labels, train_all_preds)
    train_recall = recall_score(train_all_labels, train_all_preds)
    train_f1 = f1_score(train_all_labels, train_all_preds)

    train_losses.append(running_loss / len(train_loader))
    train_accuracies.append(train_accuracy)
    train_precisions.append(train_precision)
    train_recalls.append(train_recall)
    train_f1_scores.append(train_f1)

    print(f'Epoch [{e+1}/{epoch}], Loss: {running_loss / len(train_loader):.4f}')
    print(f'Accuracy: {train_accuracy:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1 Score: {train_f1:.4f}')

    # 検証
    model.eval()
    val_loss = 0.0
    all_val_labels = []
    all_val_preds = []

    with torch.no_grad():
      for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        val_loss += loss.item()

        _, preds = torch.max(outputs, 1)
        all_val_labels.extend(labels.cpu().numpy())
        all_val_preds.extend(preds.cpu().numpy())

    val_accuracy = (torch.tensor(all_val_preds) == torch.tensor(all_val_labels)).sum().item() / len(all_val_labels)
    val_precision = precision_score(all_val_labels, all_val_preds)
    val_recall = recall_score(all_val_labels, all_val_preds)
    val_f1 = f1_score(all_val_labels, all_val_preds)

    val_losses.append(val_loss / len(val_loader))
    val_accuracies.append(val_accuracy)
    val_precisions.append(val_precision)
    val_recalls.append(val_recall)
    val_f1_scores.append(val_f1)

    print(f'Validation Loss: {val_loss / len(val_loader):.4f}')
    print(f'Validation Accuracy: {val_accuracy:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1 Score: {val_f1:.4f}')

    if (e + 1) % 5 == 0:
      output_model_path = model_path.format(epoch=e+1)
      torch.save(model.state_dict(), output_model_path)

  # モデルの保存
  output_model_path = model_path.format(epoch=epoch)
  torch.save(model.state_dict(), output_model_path)

  plot_loss(epoch, train_losses, val_losses)
  plot_accuracy(epoch, train_accuracies, val_accuracies)
  plot_precision(epoch, train_precisions, val_precisions)
  plot_recall(epoch, train_recalls, val_recalls)
  plot_f1_score(epoch, train_f1_scores, val_f1_scores)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-c', '--config', help='Select YML File')

  args = parser.parse_args()

  main(args.config)