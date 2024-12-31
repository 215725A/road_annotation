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