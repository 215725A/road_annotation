import csv
import numpy as np
import pandas as pd

from .setup import create_dir

def load_result_data(csv_path):
  info = {}

  with open(csv_path, mode='r') as f:
    reader = csv.reader(f)
    for row in reader:
      if row:
        if row[0] == 'Slope':
          info['Slope'] = float(row[1])
        elif row[0] == 'Intercept':
          info['Intercept'] = float(row[1])

  return info


def load_datas(csv_path):
  df = pd.read_csv(csv_path, skipfooter=1, engine='python')

  predict = df['Predict']
  actual = df['Actual']
  road_area = df['Road_Area']

  return predict, actual, road_area


def output_car_area_correlation_csv(car_counts, road_areas, correlation, slope, intercept):
  output_path = '../data/csv/result_data.csv'
  create_dir(output_path)

  with open(output_path, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(['car_num', 'road_area'])
    for car_count, road_area in zip(car_counts, road_areas):
      writer.writerow([car_count, road_area])
    writer.writerow(["Correlation", correlation])
    writer.writerow(['Slope', slope])
    writer.writerow(['Intercept', intercept])


def output_mae_csv(predict_car_counts, actual_car_counts, total_road_areas, output_csv_path):
  create_dir(output_csv_path)

  mae = np.mean(np.abs(np.array(actual_car_counts) - np.array(predict_car_counts)))
  print(f"Mean Absolute Error (MAE): {mae}")

  with open(output_csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Predict', 'Actual', 'Road_Area'])
    for predict, actual, total_road_area in zip(predict_car_counts, actual_car_counts, total_road_areas):
      writer.writerow([predict, actual, total_road_area])
    writer.writerow(['MAE', mae])


def output_car_pixel_mean(car_px_mean):
  output_path = f'../data/csv/car_mean.csv'
  create_dir(output_path)

  with open(output_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['car_px_mean(px)', car_px_mean])


def output_congestion_rate(predict_congestion, actual_congestion):
  output_path = '../data/csv/congestion.csv'
  create_dir(output_path)

  with open(output_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Predict', 'Actual'])
    for predict, actual in zip(predict_congestion, actual_congestion):
      writer.writerow([predict, actual])