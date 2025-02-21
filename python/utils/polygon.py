import cv2
import numpy as np

import pandas as pd

from .calc import calc_weight, calc_line

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

  return (x, y)

def calc_road_area(frame, name):
  frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  green_mask = (frame_rgb[:, :, 0] == 0) & (frame_rgb[:, :, 1] == 255) & (frame_rgb[:, :, 2] == 0)
  red_mask = (frame_rgb[:, :, 0] == 255) & (frame_rgb[:, :, 1] == 0) & (frame_rgb[:, :, 2] == 0)

  rows_red_count = np.sum(red_mask, axis=1)  # number of red pixels
  rows_green_count = np.sum(green_mask, axis=1)  # number of green pixels

  green_area = green_mask.sum()
  total_area = frame_rgb.shape[0] * frame_rgb.shape[1]
  green_area_ratio = green_area / total_area * 100

  print("=====================================")
  print(f'Road Area: {green_area}')
  print(f'Total Area: {total_area}')
  print(f'Road Ratio: {green_area_ratio}')
  print("=====================================")

  result = pd.DataFrame({"Row": range(len(green_mask)), "Red Pixels": rows_red_count, "Green Pixels": rows_green_count})
  result.to_csv(f"row_pixel_counts_for_calc_area_{name}.csv", index=False)

  return green_area


def calc_road_area2(frame, cropped_image):
  cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
  green_mask = (cropped_image[:, :, 0] == 0) & (cropped_image[:, :, 1] == 255) & (cropped_image[:, :, 2] == 0)
  red_mask = (cropped_image[:, :, 0] == 255) & (cropped_image[:, :, 1] == 0) & (cropped_image[:, :, 2] == 0)

  rows_red_count = np.sum(red_mask, axis=1)  # number of red pixels
  rows_green_count = np.sum(green_mask, axis=1)  # number of green pixels

  max_pixels_width = (rows_red_count + rows_green_count).max()

  weights = calc_weight(rows_red_count, rows_green_count, max_pixels_width)

  total_area = frame.shape[0] * frame.shape[1]

  green_area = rows_green_count * weights
  green_area = green_area.sum()

  green_area_ratio = green_area / total_area * 100

  print("=====================================")
  print(f'Road Area: {green_area}')
  print(f'Total Area: {total_area}')
  print(f'Road Ratio: {green_area_ratio}')
  print("=====================================")

  result = pd.DataFrame({"Row": range(len(rows_red_count)), "Red Pixels": rows_red_count, "Green Pixels": rows_green_count})
  result.to_csv("row_pixel_counts_for_calc_area.csv", index=False)

  return green_area


def pixels_info(frame):
  # Change BGR to RGB
  image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

  # Create mask for green, red
  red_mask = (image_rgb[:, :, 0] == 255) & (image_rgb[:, :, 1] == 0) & (image_rgb[:, :, 2] == 0)
  green_mask = (image_rgb[:, :, 0] == 0) & (image_rgb[:, :, 1] == 255) & (image_rgb[:, :, 2] == 0)

  # Calculate the number of red or green pixels for each row
  rows_red_count = np.sum(red_mask, axis=1)  # number of red pixels
  rows_green_count = np.sum(green_mask, axis=1)  # number of green pixels
  

  result = pd.DataFrame({"Row": range(len(rows_red_count)), "Red Pixels": rows_red_count, "Green Pixels": rows_green_count})
  result.to_csv("row_pixel_counts.csv", index=False)


def process_cropping(frame, frame_num, points, road_areas, car_counts, output_detect_image_path):
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

  output_image_path = output_detect_image_path.format(frame_num=frame_num)

  cv2.imwrite(output_image_path, copy_frame)

  cv2.imshow('Result', copy_frame)

  cv2.waitKey(0)
  cv2.destroyAllWindows()

  # pixels_info(copy_frame)

  road_area = calc_road_area(frame, copy_frame)

  car_count = int(input('Please type car count: '))

  road_areas.append(road_area)
  car_counts.append(car_count)


def process_rower_crop(frame, points):
  frame_copy = frame.copy()

  height, width = frame_copy.shape[:2]

  point1, point2 = points
  # m, b = calc_line(point2, point1)

  # mask = np.zeros((height, width), dtype=np.uint8)

  # for x in range(width):
  #   y_line = int(m * x + b)
  #   mask[y_line:, x] = 255
  
  # lower_region = cv2.bitwise_and(frame_copy, frame_copy, mask=mask)

  y_min = min(point1[1], point2[1])

  # points_src = np.float32([
  #   point2,
  #   point1,
  #   [0, height],
  #   [width, height]
  # ])

  # points_dst = np.float32([
  #   [point2[0], y_min],
  #   [point1[0], y_min],
  #   [0, height],
  #   [width, height]
  # ])

  # matrix = cv2.getPerspectiveTransform(points_src, points_dst)

  # warped_image = cv2.warpPerspective(lower_region, matrix, (width, height - y_min))
  # warped_image = cv2.warpPerspective(lower_region, matrix, (width, height))
  # warped_image = cv2.warpPerspective(frame_copy, matrix, (width, height - y_min))
  # warped_image = cv2.warpPerspective(frame_copy, matrix, (width, height))
  # lower_region = warped_image[y_min:height, 0:width]

  # extra_height = height - y_min

  frame_lower = frame_copy[y_min:height, 0:width]

  # cv2.imshow('Lower', lower_region)
  # cv2.waitKey(0)
  # cv2.destroyWindow('Lower')

  # return lower_region, extra_height
  return frame_lower


def process_projection(frame, frame_num, points, output_detect_image_path):
  if len(points) != 4:
    print("Error: Exactly 4 points must be selected to process the polygon.")
    return

  frame_copy = frame.copy()
  height, width = frame_copy.shape[:2]

  points_dst = np.float32([
    [0, 0],       # 左上
    [width, 0],   # 右上
    [0, height],  # 左下
    [width, height]  # 右下
  ])

  matrix = cv2.getPerspectiveTransform(points, points_dst)

  warped_image = cv2.warpPerspective(frame, matrix, (width, height))

  output_image_path = output_detect_image_path.format(frame_num=frame_num)

  cv2.imwrite(output_image_path, warped_image)
  cv2.imshow('Result', warped_image)
  cv2.waitKey(0)
  cv2.destroyWindow('Result')

  return warped_image


def process_combine(higher_crop, rower_crop, frame_num, extra_height):
  height, width = higher_crop.shape[:2]
  frame_combined = np.zeros((height + extra_height, width, 3), dtype=np.uint8)

  frame_combined[:720, :] = higher_crop
  frame_combined[720:, :] = rower_crop

  cv2.imshow('Combined', frame_combined)
  cv2.waitKey(0)
  cv2.destroyWindow('Combined')


def crop_divider(frame, divider_points):
  frame_copy = frame.copy()

  mask = np.zeros(frame_copy.shape[:2], dtype=np.uint8)

  cv2.fillPoly(mask, [np.array(divider_points, dtype=np.int32)], 255)

  masked_frame = cv2.bitwise_and(frame_copy, frame_copy, mask=mask)

  return masked_frame


def car_objects(frame, points, car_points):
  width, height = frame.shape[:2]

  mask_src = np.zeros((height, width), dtype=np.uint8)

  src_polygon = np.array(points, dtype=np.int32)

  cv2.fillPoly(mask_src, [src_polygon], color=255)

  src_points = np.array(points, dtype=np.float32)

  dst_points = np.float32([
    [0, 0],       # 左上
    [width, 0],   # 右上
    [0, height],  # 左下
    [width, height]  # 右下
  ])

  M = cv2.getPerspectiveTransform(src_points, dst_points)
  dst_size = (width, height)

  mask_dst = cv2.warpPerspective(mask_src, M, dst_size)

  mask_roi = np.zeros((dst_size[1], dst_size[0]), dtype=np.uint8)
  for car_point in car_points:
    x, y, w, h = car_point
    roi_pts = np.array([
      [x, y],
      [x+w, y],
      [x+w, y+h],
      [x, y+h]
    ], dtype=np.float32).reshape(-1, 1, 2)
    roi_pts_transformed = cv2.perspectiveTransform(roi_pts, M)
    roi_pts_transformed = roi_pts_transformed.reshape(-1, 2).astype(np.int32)
    cv2.fillPoly(mask_roi, [roi_pts_transformed], color=255)

  intersection_mask = cv2.bitwise_and(mask_dst, mask_roi)
  object_pixel_count = cv2.countNonZero(intersection_mask)

  return object_pixel_count