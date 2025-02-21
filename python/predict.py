import argparse
import os
import cv2

from utils.yaml import load_config
from utils.csv import load_result_data, output_mae_csv
from utils.click_event import select_points
from utils.polygon import find_intersection, process_rower_crop, process_projection, calc_road_area
from utils.calc import calc_line, calc_new_point
from utils.setter import set_rect_point

def main(yaml_path):
  config = load_config(yaml_path)
  csv_path = config['csv_path']
  val_image_path = config['val_image_path']
  output_csv_path = config['output_csv_path']
  output_detect_image_path = config['output_detect_image_path']

  info = load_result_data(csv_path)

  slope = info['Slope']
  intercept = info['Intercept']

  val_images = sorted(os.listdir(val_image_path))

  predict_car_counts = []
  actual_car_counts = []
  total_road_areas = []

  for image in val_images:
    points = []
    frame_path = f'{val_image_path}/{image}'
    frame_num = image[7:-4]

    frame = cv2.imread(frame_path)

    params = {'frame': frame, 'points': points}
    cv2.imshow("Select 4 points", frame)
    cv2.setMouseCallback("Select 4 points", select_points, param=params)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("#####################################")
    print(f"Frame: {frame_num}")
    print("#####################################")

    intersection = find_intersection(points)
    line1 = calc_line(points[0], points[1])
    line2 = calc_line(points[2], points[3])

    under_two_points = sorted(points, key=lambda x:x[1], reverse=True)[:2]
    # rower_crop, extra_height = process_rower_crop(frame, under_two_points)
    frame_lower = process_rower_crop(frame, under_two_points)

    left_point = calc_new_point(intersection, line1)
    right_point = calc_new_point(intersection, line2)

    rect_points = set_rect_point(intersection, left_point, right_point, points)

    higher_crop = process_projection(frame, frame_num, rect_points, output_detect_image_path)

    top_road_area = calc_road_area(higher_crop, 'top')
    low_road_area = calc_road_area(frame_lower, 'bottom')

    total_road_area = top_road_area + low_road_area

    print("-------------------------------------")
    print(f'Total Road Area: {total_road_area}')
    print("-------------------------------------")

    predict_car_count = int(slope * total_road_area + intercept)
    actual_value = int(input('Please type car count: '))
    predict_car_counts.append(predict_car_count)
    actual_car_counts.append(actual_value)
    total_road_areas.append(total_road_area)

  output_mae_csv(predict_car_counts, actual_car_counts, total_road_areas, output_csv_path)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-c', '--config', help='Select YML File')

  args = parser.parse_args()

  main(args.config)