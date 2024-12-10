import yaml

def load_config(yaml_path):
  with open(yaml_path, 'r') as yml:
    config = yaml.safe_load(yml)
  
  return config