import argparse

import cv2

from utils.setup import create_dir
from utils.yaml import load_config
from utils.click_event import select_points
from utils.polygon import find_intersection, process_cropping

def main(yaml_path):
  config = load_config(yaml_path)

  segmented_image_path = config['segmented_image_path']
  output_detect_image_path = config['output_detect_image_path']

  create_dir(output_detect_image_path)

  frame = cv2.imread(segmented_image_path)
  points = []
  params = {'frame': frame, 'points': points}

  cv2.imshow("Select 4 points", frame)
  cv2.setMouseCallback("Select 4 points", select_points, param=params)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

  x, y = find_intersection(points)

  process_cropping(frame, points, output_detect_image_path)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-c', '--config', help='Select YML File')

  args = parser.parse_args()

  main(args.config)