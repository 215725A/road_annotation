import argparse
import multiprocessing

from utils.setup import create_dir
from utils.yaml import load_config

from validation.dataset import Dataset


def main(yaml_path):
  config = load_config(yaml_path)
  model_path = config['output_model_path']
  video_path = config['video_path']
  output_video_path = config['output_video_path']
  output_frame_path = config['output_frame_path']
  patch_size = config['patch_size']

  create_dir(output_video_path)

  model_path = model_path.format(epoch=100)

  # モデルのロード（学習済みモデルをロード）
  model = Dataset(model_path, video_path, output_video_path, output_frame_path, patch_size)
  
  # model.run()  # process for video
  # model.run2() # process for only one frame
  model.run3() # process for few frame


if __name__ == '__main__':
  multiprocessing.set_start_method('spawn')

  parser = argparse.ArgumentParser()
  parser.add_argument('-c', '--config', help='Select YML File')

  args = parser.parse_args()

  main(args.config)