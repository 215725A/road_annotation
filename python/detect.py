import argparse
import os

import cv2

from utils.setup import create_dir
from utils.yaml import load_config
from utils.click_event import select_points
from utils.polygon import find_intersection, process_cropping, process_projection, process_rower_crop, process_combine, calc_road_area, crop_divider
from utils.plot import plot_car_area, plot_least_square
from utils.calc import calc_line, calc_new_point, pearson_correlation, least_squares, calc_divider_area
from utils.setter import set_rect_point, set_plot_info
from utils.csv import output_car_area_correlation_csv

def main(yaml_path):
  config = load_config(yaml_path)

  segmented_image_path = config['segmented_image_path']
  segmented_images = sorted(os.listdir(segmented_image_path))
  output_detect_image_path = config['output_detect_image_path']

  create_dir(output_detect_image_path)

  road_areas = []
  car_counts = []

  for image in segmented_images:
    points = []
    divider_points = []
    frame_path = f'{segmented_image_path}/{image}'
    frame_num = image[7:-4]

    frame = cv2.imread(frame_path)

    params = {'frame': frame, 'points': points}
    cv2.imshow("Select 4 points", frame)
    cv2.setMouseCallback("Select 4 points", select_points, param=params)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # process_cropping(frame, frame_num, points, road_areas, car_counts, output_detect_image_path)
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

    print("#####################################")
    print(f'Information: Frame {frame_num}')
    print("#####################################")


    top_road_area = calc_road_area(higher_crop, 'top')
    low_road_area = calc_road_area(frame_lower, 'bottom')

    total_road_area = top_road_area + low_road_area

    print("-------------------------------------")
    print(f'Total Road Area: {total_road_area}')
    print("-------------------------------------")

    set_plot_info(total_road_area, road_areas, car_counts)

    result_frame_path = output_detect_image_path.format(frame_num=frame_num)
    result_frame = cv2.imread(result_frame_path)
    params = {'frame': result_frame, 'divider_points': divider_points}

    cv2.imshow("Select divider points", result_frame)
    cv2.setMouseCallback("Select divider points", select_points, param=params)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cropped_divider = crop_divider(result_frame, divider_points)

    divider_area = calc_divider_area(cropped_divider)

    # combined_frame = process_combine(higher_crop, rower_crop, frame_num, extra_height)

  plot_car_area(road_areas, car_counts)
  correlation = pearson_correlation(car_counts, road_areas)

  if correlation > 0:
    relation = "正の相関があります。"
  elif correlation < 0:
    relation = "負の相関があります。"
  else:
    relation = "相関がありません。"
  
  print(correlation)
  print(relation)

  slope, intercept = least_squares(road_areas, car_counts)
  str_approximation = f'Approximation: {slope:.6f}x + {intercept:.2f}'
  print(str_approximation)

  output_car_area_correlation_csv(car_counts, road_areas, correlation, slope, intercept)

  plot_least_square(road_areas, car_counts, slope, intercept)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-c', '--config', help='Select YML File')

  args = parser.parse_args()

  main(args.config)