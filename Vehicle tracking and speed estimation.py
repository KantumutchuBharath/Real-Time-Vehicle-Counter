import cv2
import numpy as np
import time

import os

video_path = 'Video.mp4'
if not os.path.exists(video_path):
    raise FileNotFoundError(f"Video file not found: {video_path}")

cap = cv2.VideoCapture(video_path)
count_line_pos = 550
min_width_rectangle = 80
min_height_rectangle = 80
slowdown = 30

algo = cv2.bgsegm.createBackgroundSubtractorMOG()

def center_handle(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy

detect = []
offset = 5
count = 0
vehicle_id = 0
vehicle_centers = {}

while True:
    ret, frame1 = cap.read()
    if not ret:
        print("Failed to read frame from video. Exiting...")
        break
    gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 5)

    img_sub = algo.apply(blur)
    dilat = cv2.dilate(img_sub, np.ones((5, 5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
    dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)
    counter, _ = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.line(frame1, (25, count_line_pos), (1200, count_line_pos), (255, 127, 0), 3)

    new_vehicle_centers = []
    for (i, c) in enumerate(counter):
        (x, y, w, h) = cv2.boundingRect(c)
        validate_counter = (w >= min_width_rectangle) and (h >= min_height_rectangle)
        if not validate_counter:
            continue

        center = center_handle(x, y, w, h)
        new_vehicle_centers.append(center)
        cv2.circle(frame1, center, 4, (0, 50, 255), -1)

        closest_id = None
        closest_distance = 30 
        for vid, prev_center in vehicle_centers.items():
            distance = np.linalg.norm(np.array(center) - np.array(prev_center))
            if distance < closest_distance:
                closest_distance = distance
                closest_id = vid

        if closest_id is None:
            vehicle_id += 1
            closest_id = vehicle_id

        vehicle_centers[closest_id] = center

        for (cx, cy) in detect:
            if cy < (count_line_pos + offset) and cy > (count_line_pos - offset):
                count += 1
                speed_text = f"Vehicle {count}"
                text_x = x
                text_y = y - 10 if y - 10 > 20 else y + h + 20
                cv2.putText(frame1, speed_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2) 
                print(f"Vehicle {count}")
                cv2.line(frame1, (25, count_line_pos), (1200, count_line_pos), (0, 127, 255), 3)
                detect.remove((cx, cy))

    detect = new_vehicle_centers  

    cv2.putText(frame1, f"VEHICLE COUNTER: {count}", (350, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 7)

    cv2.imshow('Video Original', frame1)

    key = cv2.waitKey(slowdown)
    if key == 13:
        break

cv2.destroyAllWindows()
cap.release()
