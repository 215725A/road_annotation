import cv2
import pandas as pd


def set_label():
  print("Please select number")
  print("1: Road(True)")
  print("2: Not Road(False)")
  label_input = input("Select (1/2): ")

  if label_input == '1':
    return True
  elif label_input == '2':
    return False
  else:
    print("Invalid choice. Selected 'false' as the default")
    return False


def click_event(event, x, y, flags, param):
  frame = param['frame']
  output_image_path = param['output_image_path']
  output_csv_path = param['output_csv_path']
  max_block_size = param['max_block_size']

  first_label = None

  if event == cv2.EVENT_LBUTTONDOWN:
    print(f'Clicked position: ({x}, {y})')

    height, width = frame.shape[:2]

    block_sizes = [i for i in range(3, max_block_size+1, 2)]

    for block_size in block_sizes:
      half_block = block_size // 2

      start_x = max(x - half_block, 0)
      start_y = max(y - half_block, 0)

      end_x = min(x + half_block + 1, width)
      end_y = min(y + half_block + 1, height)

      block = frame[start_y:end_y, start_x:end_x]

      if block_size == 3 and first_label is None:
        cv2.imshow("Extracted Block", block)
        cv2.waitKey(1000)
        first_label = set_label()

      label = first_label

      cv2.imshow(f'Extracted Block: {block_size}*{block_size}', block)

      output_filename = f'{output_image_path}extracted_block_{block_size}_{x}_{y}.png'
      cv2.imwrite(output_filename, block)

      output_csv_filename = f'{output_csv_path}labels.csv'
      label_data = pd.DataFrame([[output_filename, label]], columns=['image_path', 'label'])
      label_data.to_csv(output_csv_filename, mode='a', header=False, index=False)

      print(f"Image: {output_filename} and label: '{label}' saved")

      if block_size != max_block_size:
        print("Press the key to proceed to the next image")

    cv2.destroyAllWindows()


def select_points(event, x, y, flags, param):
  frame = param['frame']
  points = param['points']

  if event == cv2.EVENT_LBUTTONDOWN:
    points.append((x, y))
    print(f'Point {len(points)} selected: ({x}, {y})')

  if event == cv2.EVENT_RBUTTONDOWN and len(points) > 0:
    del points[-1]

  temp_img = frame.copy()
  h, w = temp_img.shape[0], temp_img.shape[1]
  cv2.line(temp_img, (x, 0), (x, h), (255, 255, 255), 1)
  cv2.line(temp_img, (0, y), (w, y), (255, 255, 255), 1)

  for i in range(len(points)):
    cv2.circle(temp_img, (points[i][0], points[i][1]), 3, (128, 128, 128), 3)

    if 0 < i:
      cv2.line(temp_img, (points[i][0], points[i][1]),
      (points[i-1][0], points[i-1][1]), (0, 0, 0), 2)

  cv2.imshow('Select 4 points', temp_img)


def draw_rois(frame, roi_list):
  for (x, y, w, h) in roi_list:
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)


def select_car(event, x, y, flags, param):
  frame = param['frame']
  rois = param['rois']

  if event == cv2.EVENT_LBUTTONDOWN:
    param['drawing'] = True
    param['ix'], param['iy'] = x, y

  elif event == cv2.EVENT_MOUSEMOVE:
    if param.get('drawing', False):
      frame_copy = frame.copy()
      draw_rois(frame_copy, rois)
      ix, iy = param['ix'], param['iy']
      cv2.rectangle(frame_copy, (ix, iy), (x, y), (0, 255, 0), 2)
      param['frame_copy'] = frame_copy

  elif event == cv2.EVENT_LBUTTONUP:
    if param.get('drawing', False):
      param['drawing'] = False
      ix, iy = param['ix'], param['iy']

      x0, y0 = min(ix, x), min(iy, y)
      w, h = abs(x - ix), abs(y - iy)
      if w > 0 and h > 0:
        rois.append((x0, y0, w, h))

      frame_copy = frame.copy()
      draw_rois(frame_copy, rois)
      param['frame_copy'] = frame_copy

    print("Finished")
    print('===========================')

  return rois