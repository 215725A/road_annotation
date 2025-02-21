import os
import cv2
import argparse
import numpy as np

from utils.yaml import load_config
from utils.click_event import select_points, select_car, draw_rois
from utils.polygon import find_intersection, calc_line, car_objects, process_rower_crop, process_projection, calc_road_area
from utils.calc import calc_new_point, max_car_counts, congestion_rate
from utils.csv import output_car_pixel_mean, load_datas, output_congestion_rate
from utils.setter import set_rect_point
from utils.plot import plot_congestion_rate

def main(yaml_path):
  config = load_config(yaml_path)
  config_image_path = config['output_image_path']
  output_csv_path = config['output_csv_path']
  car_height = config['car_height']
  car_width = config['car_width']
  road_height = config['road_height']
  road_width = config['road_width']
  divider_height = config['divider_height']
  divider_width = config['divider_width']

  image_path = f'{config_image_path}'
  all_files = sorted(os.listdir(image_path))
  image_files = [f for f in all_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

  car_rois = []

  sum_car_px = 0

  rois = []
  points = []
  frame_path = f'{config_image_path}frame_0001.png'
  frame = cv2.imread(frame_path)

  params = {
            'frame': frame, 
            'frame_copy': frame.copy(),
            'rois': rois, 
            'drawing': False,
            'ix': 0, 
            'iy': 0
          }

  window_name = 'Select Car ROI'

  # cv2.imshow(window_name, frame)
  cv2.namedWindow(window_name)
  cv2.setMouseCallback(window_name, select_car, param=params)

  while True:
    if 'frame_copy' in params:
      display_frame = params['frame_copy']
    else:
      display_frame = frame.copy()
    cv2.imshow(window_name, display_frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
      break
    elif key == ord('c'):
      if rois:
        rois.pop()
        temp = frame.copy()
        draw_rois(temp, rois)
        params['frame_copy'] = temp
    elif key == ord('r'):
      rois = []
      temp = frame.copy()
      params['frame_copy'] = temp

  cv2.destroyAllWindows()

  params = {'frame': frame, 'points': points}
  cv2.imshow('Select 4 points', frame)
  cv2.setMouseCallback("Select 4 points", select_points, param=params)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

  intersection = find_intersection(points)
  under_two_points = sorted(points, key=lambda x:x[1], reverse=True)[:2]

  line1 = calc_line(points[0], points[1])
  line2 = calc_line(points[2], points[3])

  left_point = calc_new_point(intersection, line1)
  right_point = calc_new_point(intersection, line2)

  projection_pts = set_rect_point(intersection, left_point, right_point, points)

  car_pixels = car_objects(frame, projection_pts, rois)
  print(f'Car Pixels: {car_pixels} px')

  sum_car_px += car_pixels
  car_rois += rois

  mean_car_px = sum_car_px // len(car_rois)

  print(f'Car pixels mean: {mean_car_px} px')

  output_car_pixel_mean(mean_car_px)

  predict_cars, actual_cars, road_areas = load_datas(output_csv_path)

  predict_congestion = []
  actual_congestion = []
  pre_max_car_counts = []

  act_road_area = (road_height * road_width) - (divider_height * divider_width)
  act_max_car_cnt = max_car_counts(act_road_area, (car_height * car_width))

  for p, a, r in zip(predict_cars, actual_cars, road_areas):
    pre_max_car_cnt = max_car_counts(r + mean_car_px * a, mean_car_px)

    pre_congestion = congestion_rate(p, pre_max_car_cnt)
    act_congestion = congestion_rate(a, act_max_car_cnt)

    predict_congestion.append(pre_congestion)
    actual_congestion.append(act_congestion)
    pre_max_car_counts.append(pre_max_car_cnt)

  print(f'Predict car max: {sum(pre_max_car_counts) // len(pre_max_car_counts)}')
  print(f'Actual car max: {act_max_car_cnt}')
  plot_congestion_rate(predict_congestion, actual_congestion)
  output_congestion_rate(predict_congestion, actual_congestion)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-c', '--config', help='Select YML File')

  args = parser.parse_args()

  main(args.config)