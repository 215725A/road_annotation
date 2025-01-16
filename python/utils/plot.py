import matplotlib.pyplot as plt
from .setup import create_dir 


def plot_loss(epochs, train_losses, val_losses):
  fig_path = f'../data/plots/loss.png'
  create_dir(fig_path)

  epoch = [i for i in range(1, epochs+1)]

  plt.plot(epoch, train_losses, label='Train Loss', marker='o')
  plt.plot(epoch, val_losses, label='Val Loss', marker='x')
  plt.title('Loss per epoch')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.legend()
  plt.grid()
  plt.savefig(fig_path, bbox_inches='tight')
  plt.show()

def plot_accuracy(epochs, train_accuracies, val_accuracies):
  fig_path = f'../data/plots/acc.png'
  create_dir(fig_path)

  epoch = [i for i in range(1, epochs+1)]

  plt.plot(epoch, train_accuracies, label='Train Accuracy', marker='o')
  plt.plot(epoch, val_accuracies, label='Val Accuracy', marker='x')
  plt.title('Accuracy per epoch')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.legend()
  plt.grid()
  plt.savefig(fig_path, bbox_inches='tight')
  plt.show()

def plot_precision(epochs, train_precisions, val_precisions):
  fig_path = f'../data/plots/precision.png'
  create_dir(fig_path)

  epoch = [i for i in range(1, epochs+1)]

  plt.plot(epoch, train_precisions, label='Train Precision', marker='o')
  plt.plot(epoch, val_precisions, label='Val Precisions', marker='x')
  plt.title('Precision per epoch')
  plt.xlabel('Epoch')
  plt.ylabel('Precision')
  plt.legend()
  plt.grid()
  plt.savefig(fig_path, bbox_inches='tight')
  plt.show()

def plot_recall(epochs, train_recalls, val_recalls):
  fig_path = f'../data/plots/recall.png'
  create_dir(fig_path)

  epoch = [i for i in range(1, epochs+1)]

  plt.plot(epoch, train_recalls, label='Train Recall', marker='o')
  plt.plot(epoch, val_recalls, label='Val Recall', marker='x')
  plt.title('Recall per epoch')
  plt.xlabel('Epoch')
  plt.ylabel('Recall')
  plt.legend()
  plt.grid()
  plt.savefig(fig_path, bbox_inches='tight')
  plt.show()

def plot_f1_score(epochs, train_f1_scores, val_f1_scores):
  fig_path = f'../data/plots/f1_score.png'
  create_dir(fig_path)

  epoch = [i for i in range(1, epochs+1)]

  plt.plot(epoch, train_f1_scores, label='Train F1 Score', marker='o')
  plt.plot(epoch, val_f1_scores, label='Val F1 Score', marker='x')
  plt.title('F1 Score per epoch')
  plt.xlabel('Epoch')
  plt.ylabel('F1 Score')
  plt.legend()
  plt.grid()
  plt.savefig(fig_path, bbox_inches='tight')
  plt.show()