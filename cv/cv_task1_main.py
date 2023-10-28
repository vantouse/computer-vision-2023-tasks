import cv2
import numpy as np
import time


def on_trackbar(*args):
    pass


# capture camera input
capture_cam = cv2.VideoCapture(0)
default_size = [640, 480]

# settings window
cv2.namedWindow('settings')
cv2.createTrackbar('trackbar_mode', 'settings', 0, 1, on_trackbar)
cv2.createTrackbar('H_lower', 'settings', 0, 179, on_trackbar)
cv2.createTrackbar('S_lower', 'settings', 0, 255, on_trackbar)
cv2.createTrackbar('V_lower', 'settings', 0, 255, on_trackbar)
cv2.createTrackbar('H_upper', 'settings', 179, 179, on_trackbar)
cv2.createTrackbar('S_upper', 'settings', 255, 255, on_trackbar)
cv2.createTrackbar('V_upper', 'settings', 255, 255, on_trackbar)
cv2.moveWindow('settings', 0, 480)
trackbar_mode = 0

time_init = time.time_ns() // (10 ** 9)
time_prev = time.time_ns() // (10 ** 9)
fps = list()

while capture_cam.isOpened():
    # timer
    time_curr = time.time_ns() // (10 ** 9)
    if time_curr - time_prev >= 1:
        time_prev = time_curr
        print(len(fps))
        fps.clear()
        # print(time_curr - time_init)

    ret, cam_frame = capture_cam.read()
    cam_frame = cv2.resize(cam_frame, default_size)
    fps.append(cam_frame)

    cam_frame_bin = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2GRAY)
    cam_frame_hsv = cv2.blur(cv2.cvtColor(cam_frame, cv2.COLOR_BGR2HSV), (3, 3))

    # setting up threshold sensitivity with trackbars (lower/upper bounds for HSV color detection)
    trackbar_mode = cv2.getTrackbarPos('trackbar_mode', 'settings')
    if trackbar_mode == 1:
        h_lower = cv2.getTrackbarPos('H_lower', 'settings')
        s_lower = cv2.getTrackbarPos('S_lower', 'settings')
        v_lower = cv2.getTrackbarPos('V_lower', 'settings')
        h_upper = cv2.getTrackbarPos('H_upper', 'settings')
        s_upper = cv2.getTrackbarPos('S_upper', 'settings')
        v_upper = cv2.getTrackbarPos('V_upper', 'settings')
        h_min = np.array((h_lower, s_lower, v_lower), np.uint8)
        h_max = np.array((h_upper, s_upper, v_upper), np.uint8)
        cam_frame_bin = cv2.inRange(cam_frame_hsv, h_min, h_max)
        thresh = cv2.threshold(cam_frame_bin, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        cv2.imshow('cam_frame_bin', thresh)
        cv2.moveWindow('cam_frame_bin', 640, 0)
    else:
        h_min = np.array((5, 110, 120), np.uint8)  # 0 110 70
        h_max = np.array((179, 200, 255), np.uint8)
        cam_frame_bin = cv2.inRange(cam_frame_hsv, h_min, h_max)
        thresh = cv2.threshold(cam_frame_bin, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # find contours and build bounding box for each contour
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    for contour in contours:
        if cv2.arcLength(contour, True) > 100:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(cam_frame, (x, y), (x + w, y + h), (0, 255, 0), 5, 1)

    # camera preview
    cv2.imshow('cam_frame', cam_frame)
    cv2.moveWindow('cam_frame', 0, 0)

    # exit on 'Q' key pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture_cam.release()
cv2.destroyAllWindows()
