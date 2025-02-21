import numpy as np

def set_rect_point(intersection, left_point, right_point, points):
  below_points = [point for point in points if point[1] > intersection[1]]
  # below_points.reverse()

  rect_point = [left_point] + [right_point] + below_points[:2]

  return np.array(rect_point, dtype=np.float32)

def set_plot_info(total_road_area, road_areas, car_counts):
  car_count = int(input('Please type car count: '))

  road_areas.append(total_road_area)
  car_counts.append(car_count)