import math
import cv2
import numpy as np

from decimal import Decimal, ROUND_HALF_UP

def calc_weight(rows_red_count, rows_green_count, max_pixels_width):
  weights = []
  for red_pixel, green_pixel in zip(rows_red_count, rows_green_count):
    sum_pixels = red_pixel + green_pixel
    if sum_pixels == 0:
      weights.append(0)
      continue

    weights.append(max_pixels_width / (red_pixel + green_pixel))

  return np.array(weights)

def calc_line(point1, point2):
  x1, y1 = point1
  x2, y2 = point2

  if x1 == x2:
    raise ValueError("If two points have the same x-coordinate, the slope goes to infinity.")

  m = (y2 - y1) / (x2 - x1)
  b = y1 - m * x1

  return (m, b)


def calc_new_point(intersection, line_equation):
  x_c, y_c = intersection
  m, b = line_equation

  y_dash = y_c + 10
  x_dash = (y_dash - b) / m
  x_dash = Decimal(x_dash)
  x_dash = x_dash.quantize(Decimal('0'), rounding=ROUND_HALF_UP)

  return (int(x_dash), y_dash)


def pearson_correlation(x, y):
  # Calc average
  mean_x = sum(x) / len(x)
  mean_y = sum(y) / len(y)
  
  numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
  
  denominator = math.sqrt(sum((xi - mean_x) ** 2 for xi in x) * sum((yi - mean_y) ** 2 for yi in y))
  
  return numerator / denominator if denominator != 0 else 0

def least_squares(x, y):
  # Calc average
  mean_x = sum(x) / len(x)
  mean_y = sum(y) / len(y)
  
  # Find the slope
  numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
  denominator = sum((xi - mean_x) ** 2 for xi in x)
  slope = numerator / denominator
  
  # Find the intercept
  intercept = mean_y - slope * mean_x
  
  return slope, intercept


def calc_divider_area(frame):
  frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  green_mask = (frame_rgb[:, :, 0] == 0) & (frame_rgb[:, :, 1] == 255) & (frame_rgb[:, :, 2] == 0)
  red_mask = (frame_rgb[:, :, 0] == 255) & (frame_rgb[:, :, 1] == 0) & (frame_rgb[:, :, 2] == 0)

  green_area = np.sum(green_mask)
  red_area = np.sum(red_mask)


  print(f'Divider Area: {red_area}')

  return red_area


def max_car_counts(road_area, car_px_mean):
  return road_area // car_px_mean


def congestion_rate(car_count, max_car_counts):
  return car_count / max_car_counts * 100