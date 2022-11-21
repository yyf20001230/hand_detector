import cv2
import numpy as np
import math

cap = cv2.VideoCapture(1)

while True:
    success, img = cap.read()

    edge = cv2.Canny(img, 50, 200, None, 3)

    lines = cv2.HoughLines(edge, 1, np.pi / 180, 120, None, 0, 0)

    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv2.line(img, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)
    cv2.imshow("Output-Keypoints", img)

    cv2.waitKey(1)
