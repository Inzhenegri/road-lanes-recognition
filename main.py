import cv2
import numpy as np

cap = cv2.VideoCapture('videos/output2.mp4')

if not cap.isOpened():
    raise Exception('Cannot open the input video')

img_size = (360, 200)

# Координаты маски (белой трапеции)
src = np.float32([[10, 180],
                 [400, 180],
                 [325, 120],
                 [60, 120]])  # src stands for pixel vertices on an image, the first is x, the second is y
src_draw = np.array(src, np.int32)
dst = np.float32([[1, img_size[0]],
                  [img_size[1], img_size[0]],
                  [img_size[1], 1],
                  [1, 1]])

while cv2.waitKey(1) != ord('q'):
    ret, frame = cap.read()
    if not ret:
        print('The video has ended.')
        break

    resized_frame = cv2.resize(frame, img_size)

    r_channel = resized_frame[:, :, 2]
    binary = np.zeros_like(r_channel)
    binary[r_channel > 200] = 1

    hls = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2HLS)
    s_channel = resized_frame[:, :, 2]
    binary2 = np.zeros_like(s_channel)
    binary2[r_channel > 160] = 1

    all_binary = np.zeros_like(binary)
    all_binary[(binary == 1) | (binary2 == 1)] = 255
    # cv2.imshow('all_binary', all_binary)

    all_binary_visual = all_binary.copy()
    cv2.polylines(all_binary_visual, [src_draw], True, 255)
    # cv2.imshow('polygon', all_binary_visual)

    transform_matrix = cv2.getPerspectiveTransform(src, dst)
    warped_image = cv2.warpPerspective(all_binary, transform_matrix, (img_size[1], img_size[0]), flags=cv2.INTER_LINEAR)
    cv2.imshow('warped_image', warped_image)

    # Считаем самые белые пиксели в каждом столбце
    histogram = np.sum(warped_image[warped_image.shape[0] // 2:, :], axis=0)  # Суммируем по вертикали (axis=0)

    midpoint = histogram.shape[0] // 2
    # print(midpoint)
    whitest_pixel_index_left = np.argmax(histogram[:midpoint])
    whitest_pixel_index_right = np.argmax(histogram[midpoint:]) + midpoint

    # Рисуем линии на копии warped_image
    warped_image_visual = warped_image.copy()
    cv2.line(warped_image_visual,
             (whitest_pixel_index_left, 0),
             (whitest_pixel_index_left, warped_image_visual.shape[0]),
             110, 2)
    cv2.line(warped_image_visual,
             (whitest_pixel_index_right, 0),
             (whitest_pixel_index_right, warped_image_visual.shape[0]),
             110, 2)
    cv2.imshow('Warped visual', warped_image_visual)

    # Ищем белые пиксели по нарисованным столбцам
    nwindows = 9
    window_height = int(warped_image.shape[0] / nwindows)  # Считем высоту окна
    window_half_width = 25

    x_center_left_window = whitest_pixel_index_left
    x_center_right_window = whitest_pixel_index_right

    # Получаем координаты левых и правых белых пикселей и кладём их в список
    left_lane_indices = np.array([], dtype=np.int16)
    right_lane_indices = np.array([], dtype=np.int16)

    out_img = np.dstack((warped_image, warped_image, warped_image))

    nonzero = warped_image.nonzero()
    white_pixels_index_y = np.array(nonzero[0])  # Y-координаты всех белых пикселей
    white_pixels_index_x = np.array(nonzero[1])  # X-координаты всех белых пикселей

    for window in range(nwindows):
        win_y1 = warped_image.shape[0] - (window + 1) * window_height
        win_y2 = warped_image.shape[0] - window * window_height

        left_win_x1 = x_center_left_window - window_half_width
        left_win_x2 = x_center_left_window + window_half_width

        right_win_x1 = x_center_right_window - window_half_width
        right_win_x2 = x_center_right_window + window_half_width

        cv2.rectangle(out_img, (left_win_x1, win_y1), (left_win_x2, win_y2), (50 + window * 21, 0, 0), 2)
        cv2.rectangle(out_img, (right_win_x1, win_y1), (right_win_x2, win_y2), (0, 0, 50 + window * 21), 2)
        cv2.imshow('Windows', out_img)

        good_left_indices = ((white_pixels_index_y >= win_y1) & (white_pixels_index_y <= win_y2) &
                             (white_pixels_index_x >= left_win_x1) & (white_pixels_index_x <= left_win_x2)).nonzero()[0]
        good_right_indices = ((white_pixels_index_y >= win_y1) & (white_pixels_index_y <= win_y2) &
                              (white_pixels_index_x >= right_win_x1) & (white_pixels_index_x <= right_win_x2)).nonzero()[0]

        left_lane_indices = np.concatenate((left_lane_indices , good_left_indices))
        right_lane_indices = np.concatenate((right_lane_indices, good_right_indices))

        if len(good_left_indices) > 50:
            x_center_left_window = int(np.mean(white_pixels_index_x[good_left_indices]))
        if len(good_right_indices):
            x_center_right_window = int(np.mean(white_pixels_index_x[good_right_indices]))

    out_img[white_pixels_index_y[left_lane_indices], white_pixels_index_x[left_lane_indices]] = [255, 0, 0]
    out_img[white_pixels_index_y[right_lane_indices], white_pixels_index_x[right_lane_indices]] = [0, 0, 255]

    cv2.imshow('Lane', out_img)

    left_x = white_pixels_index_x[left_lane_indices]
    left_y = white_pixels_index_y[left_lane_indices]
    right_x = white_pixels_index_x[right_lane_indices]
    right_y = white_pixels_index_y[right_lane_indices]

    left_fit = np.polyfit(left_y, left_x, 2)
    right_fit = np.polyfit(right_y, right_x, 2)

    center_fit = ((left_fit + right_fit) / 2)

    for ver_index in range(out_img.shape[0]):
        gor_index = (center_fit[0]) * (ver_index ** 2) + center_fit[1] * ver_index + center_fit[2]
        cv2.circle(out_img, (int(gor_index), int(ver_index)), 2, (255, 0, 255), 1)

    cv2.imshow('Centerline', out_img)
