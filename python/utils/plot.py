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


def plot_car_area(road_areas, car_counts):
  fig_path = f'../data/plots/car_area.png'
  create_dir(fig_path)

  plt.scatter(road_areas, car_counts, color='blue', alpha=0.7, label='Car vs. Road Area')
  plt.title('Cars vs. Road Area')
  plt.xlabel('Road Area (px)')
  plt.ylabel('Number of cars')
  plt.legend()
  plt.grid()
  plt.savefig(fig_path, bbox_inches='tight')
  plt.show()

def plot_least_square(road_areas, car_counts, a, b):
  fig_path = f'../data/plots/least_square.png'
  create_dir(fig_path)

  plt.scatter(car_counts, road_areas, color="blue", label="Car vs. Road Area vs. Least Square")
  plt.title('Cars vs. Road Area vs. Least Square')

  x_vals = range(min(car_counts), max(car_counts) + 1)
  y_vals = [a * x + b for x in x_vals]

  plt.plot(x_vals, y_vals, color="red", label=f"Approximation: y = {a:.6f}x + {b:.2f}")

  plt.xlabel("Car Counts")
  plt.ylabel("Road Area(px)")
  plt.legend()
  plt.grid()
  plt.title("Linear Regression using Least Squares Method")
  plt.savefig(fig_path, bbox_inches='tight')
  plt.show()

def plot_congestion_rate(predict_congestion, actual_congestion):
  fig_path = f'../data/plots/congestion_rates.png'
  create_dir(fig_path)

  x_val = [i for i in range(1, len(predict_congestion)+1)]

  plt.plot(x_val, predict_congestion, label='Predict Congestion', marker='o', color='blue')
  plt.plot(x_val, actual_congestion, label='Actual Congestion', marker='x', color='red')
  plt.title('Compare predicted and actual congestion rates')
  plt.xlabel('frames')
  plt.ylabel('Congestion Rate(%)')
  plt.xticks(range(1, len(predict_congestion) + 1))
  plt.ylim(0, 50)
  plt.legend()
  plt.grid()
  plt.savefig(fig_path, bbox_inches='tight')
  plt.show()