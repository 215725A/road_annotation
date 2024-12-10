import argparse

import cv2

from utils.yaml import load_config
from utils.click_event import click_event
from utils.setup import create_dir

def main(yaml_path):
  config = load_config(yaml_path)
  video_path = config['video_path']
  output_image_path = config['output_image_path']
  output_csv_path = config['output_csv_path']
  max_block_size = config['max_block_size']

  create_dir(output_image_path)
  create_dir(output_csv_path)

  cap = cv2.VideoCapture(video_path)

  ret, frame = cap.read()


  if not ret:
    print("Video not loaded")
    exit()

  params = {'frame': frame, 'output_image_path': output_image_path, 'output_csv_path': output_csv_path, 'max_block_size': max_block_size}

  cv2.imshow('Image', frame)
  cv2.setMouseCallback("Image", click_event, param=params)

  cv2.waitKey(0)
  cv2.destroyAllWindows()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-c', '--config', help='Select YML File')

  args = parser.parse_args()

  main(args.config)