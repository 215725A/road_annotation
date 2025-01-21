import cv2
import numpy as np

def find_intersection(points):
  if len(points) != 4:
    print("Error: Exactly 4 points must be selected to process the polygon.")
    return None
  
  p1, p2, p3, p4 = points

  A1 = p1[1] - p2[1]
  B1 = p2[0] - p1[0]
  C1 = p1[0] * p2[1] - p2[0] * p1[1]

  A2 = p3[1] - p4[1]
  B2 = p4[0] - p3[0]
  C2 = p3[0] * p4[1] - p4[0] * p3[1]

  denominator = A1 * B2 - A2 * B1
  if denominator == 0:
    return None

  x = (B1 * C2 - B2 * C1) // denominator
  y = (A2 * C1 - A1 * C2) // denominator

  return x, y

def process_cropping(frame, points, output_detect_image_path):
  if len(points) != 4:
    print("Error: Exactly 4 points must be selected to process the polygon.")
    return

  height, width = frame.shape[:2]

  point1, point2, point3, point4 = points

  triangle1 = np.array([
    [0, 0],
    [0, point1[1]],
    [point2[0], 0]
  ], np.int32)

  triangle2 = np.array([
    [point4[0], 0],
    [width, 0],
    [width, point3[1]]
  ], np.int32)

  mask = np.zeros((height, width), dtype=np.uint8)

  cv2.fillPoly(mask, [triangle1], 255)
  cv2.fillPoly(mask, [triangle2], 255)

  copy_frame = frame.copy()
  copy_frame[mask == 255] = [0, 0, 0]

  # mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

  # cv2.fillPoly(mask, [polygon], 255)


  # result = cv2.bitwise_and(frame, frame, mask=mask)

  # cv2.imshow("Result", result)

  cv2.imwrite(output_detect_image_path, copy_frame)

  cv2.imshow('Result', copy_frame)

  cv2.waitKey(0)
  cv2.destroyAllWindows()