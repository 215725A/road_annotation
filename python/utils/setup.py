import os


def create_dir(dir_path):
  dir_name = os.path.dirname(dir_path)

  os.makedirs(dir_name, exist_ok=True)