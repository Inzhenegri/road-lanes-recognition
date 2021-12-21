import cv2
import numpy as np

cap = cv2.VideoCapture('videos/road_lanes.mp4')

if not cap.isOpened():
    raise Exception('Cannot open the input video')

img_size = (360, 200)
src = np.int32([
    [20, 200],
    [350, 250],
    [275, 120],
    [85, 120]
])
src_draw = np.array(src, np.int32)

dst = np.float32([[0, img_size[1]],
                  [img_size[0], img_size[1]],
                  [img_size[0], 0],
                  [0, 0]])

while cv2.waitKey(1) != ord('q'):
    ret, frame = cap.read()
    if not ret:
        print('The end of the file')
        break

    resized_frame = cv2.resize(frame, img_size)
    cv2.imshow('frame', resized_frame)

    # print(resized_frame)
    # print(resized_frame[:, :, 2])

    r_channel = resized_frame[:, :, 2]
    binary = np.zeros_like(r_channel)
    binary[(r_channel > 200)] = 1

    hls = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2HLS)
    s_channel = resized_frame[:, :, 2]
    binary2 = np.zeros_like(s_channel)
    binary2[(r_channel > 160)] = 1

    all_binary = np.zeros_like(binary)
    all_binary[((binary == 1) | (binary2 == 1))] = 255

    cv2.imshow('all binary', all_binary)

    all_binary_visual = all_binary.copy()
    cv2.polylines(all_binary_visual, [src_draw], True, 255)

    cv2.imshow('polylines', all_binary_visual)
