import cv2
import torch
import numpy as np

import torch.nn as nn
from torchvision import transforms
from torchvision import models

class Dataset:
  def __init__(self, model_path, video_path, output_video_path, patch_size=64):
    self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    self.model.fc = nn.Linear(self.model.fc.in_features, 2)
    
    self.model.load_state_dict(torch.load(model_path))
    self.model.eval()

    output_filename = f'{patch_size}x{patch_size}'

    self.input_video_path = video_path
    self.output_video_path = output_video_path.format(patch_size=output_filename)
    self.patch_size = patch_size

    self.transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.model = self.model.to(self.device)

  def run(self):
    cap = cv2.VideoCapture(self.input_video_path)
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(self.output_video_path, fourcc, fps, (frame_width, frame_height))

    frame_num = 0

    while cap.isOpened():
      ret, frame = cap.read()

      if not ret:
        break

      frame_num += 1
      print(f"Processing frame {frame_num}")

      # Convert frame to RGB
      image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      output_frame = np.zeros_like(frame)

      # Process the frame in patches
      for y in range(0, frame_height, self.patch_size):
        for x in range(0, frame_width, self.patch_size):
          # Extract the patch
          patch = image[y:y+self.patch_size, x:x+self.patch_size]

          # Skip small patches at the edges
          if patch.shape[0] != self.patch_size or patch.shape[1] != self.patch_size:
            continue

          # Transform the patch and make a prediction
          patch_tensor = self.transform(patch).unsqueeze(0).to(self.device)
          with torch.no_grad():
            output = self.model(patch_tensor)
            _, pred = torch.max(output, 1)

          # Map prediction to color (e.g., green for road, red for non-road)
          color = (0, 255, 0) if pred.item() == 1 else (0, 0, 255)
          output_frame[y:y+self.patch_size, x:x+self.patch_size] = color

      # Write the processed frame to the output video
      writer.write(output_frame)
    
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
