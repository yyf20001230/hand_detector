# import the necessary packages
import numpy as np
import cv2
import math
import pyautogui

from collections import deque


pyautogui.FAILSAFE = False

# define the upper and lower boundaries of the HSV pixel
# intensities to be considered 'skin'
camera = cv2.VideoCapture(1)

start_x, start_y, end_x, end_y = 0, 150, 500, 650
saturation, value = 30, 60
offset = 0

def getMaxContours(contours):
    if not contours: return False, None
    maxIndex = 0
    maxArea = 0

    for i in range(len(contours)):
        cnt = contours[i]
        area = cv2.contourArea(cnt)

        if area > maxArea:
            maxArea = area
            maxIndex = i

    # print(cv2.contourArea(contours[maxIndex]))
    return True, contours[maxIndex]

def calculateAngle(far, start, end):
    """Cosine rule"""
    a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
    b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
    c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
    angle = math.acos((b**2 + c**2 - a**2) / (2*b*c))
    return angle

def execute(cnt):
    if cnt == 2:
        pyautogui.press("down")
    elif cnt == 3:
        pyautogui.press("up")
    elif cnt == 4:
        pyautogui.press("left")
    elif cnt == 5:
        pyautogui.press("right")

def mouseMove(x, y):
	pyautogui.moveTo(x, y)

prev_frame_move = deque([0,0,0,0,0])

while True:

    # grab the current frame
    (grabbed, frame) = camera.read()

    cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 0)

    lower = np.array([0, saturation, value], dtype = "uint8")
    upper = np.array([20, 255, 255], dtype = "uint8")

    # calibrate saturation and value based on the average pixel intensity
    intensity = np.mean(frame[start_y:end_y, start_x:end_x])
    saturation = int(-25 * math.log(intensity) + 157 + offset)
    value = int(1.11 * saturation + 27 + offset)

    # if key is up, increase the offset. if key is down, decrease the offset
    if cv2.waitKey(1) & 0xFF == ord('w'):
        offset += 3
    elif cv2.waitKey(1) & 0xFF == ord('s'):
        offset -= 3

    cv2.putText(frame, str(offset), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


    # resize the frame, convert it to the HSV color space,
    # and determine the HSV pixel intensities that fall into
    # the speicifed upper and lower boundaries
    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    skinMask = cv2.inRange(converted, lower, upper)

    # apply a series of erosions and dilations to the mask
    # using an elliptical kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skinMask = cv2.erode(skinMask, kernel, iterations = 2)
    skinMask = cv2.dilate(skinMask, kernel, iterations = 2)
    skinMask = cv2.morphologyEx(skinMask,cv2.MORPH_OPEN,kernel)

    # blur the mask to help remove noise, then apply the
    # mask to the frame
    skinMask = cv2.GaussianBlur(skinMask, (5, 5), 0)
    skin = cv2.bitwise_and(frame, frame, mask = skinMask)

    handArea = np.zeros(skin.shape, dtype=np.uint8)
    handArea[start_y: end_y, start_x: end_x] = skin[start_y: end_y, start_x: end_x]

    grayMask = cv2.cvtColor(handArea, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(grayMask, 0, 255, 0)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        
    contourFound, maxContour = getMaxContours(contours)
    if contourFound:
        
        print(prev_frame_move)
        cv2.drawContours(skin, [maxContour], -1, (0,255,0), 3)
        convexHull_withpoints = cv2.convexHull(maxContour)
        convexHull = cv2.convexHull(maxContour, returnPoints=False)

        cv2.drawContours(skin, [convexHull_withpoints], -1, (255, 0, 0), 2)
        if len(convexHull) > 0:
            prev_frame_move.popleft()
            convexityDefects = cv2.convexityDefects(maxContour, convexHull)
            cnt = 0

            if type(convexityDefects) != type(None):
                for i in range(convexityDefects.shape[0]):
                    s, e, f, d = convexityDefects[i, 0]
                    start = tuple(maxContour[s, 0])
                    end = tuple(maxContour[e, 0])
                    far = tuple(maxContour[f, 0])
                    angle = calculateAngle(far, start, end)

                    # Ignore the defects which are small and wide
                    # Probably not fingers
                    if d > math.sqrt(cv2.contourArea(maxContour)) * 80 and angle <= math.pi/2.5:
                        cnt += 1
                        cv2.circle(skin, start, 8, [255, 0, 0], -1)


            # if there are no fingers, find the tip using the centroid
            if cnt == 0:
                M = cv2.moments(maxContour)

                if M["m00"] != 0:
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    cv2.circle(skin, (cx, cy), 8, [255, 0, 0], -1)

                    found = False
                    for i in range(len(maxContour)):
                        distance = math.sqrt((maxContour[i][0][0] - cx)**2 + (maxContour[i][0][1] - cy)**2)
                        if distance >= math.sqrt(cv2.contourArea(maxContour)):
                            found = True
                            cv2.circle(skin, (maxContour[i][0][0], maxContour[i][0][1]), 8, [255, 0, 0], -1)
                            mouseMove(3072 - maxContour[i][0][0] / (end_x - start_x) * 3072, (maxContour[i][0][1] - start_y) / (end_y - start_y) * 1920)
                            break
      
                    if not found:
                         cnt = -1
                         if sum(prev_frame_move) >= 4:
                            pyautogui.click()
                         prev_frame_move.append(0)
                    else:
                         prev_frame_move.append(1)
                else:
                    prev_frame_move.append(0)
            else:
                prev_frame_move.append(0)
                        
            execute(cnt + 1) 

            cv2.putText(frame, "number of fingers: " + str(cnt + 1), (10,100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    skin[:,:start_x] = 0
    skin[:,end_x:] = 0
    skin[:start_y,:] = 0
    skin[end_y:,:] = 0

    # show the skin in the image along with the mask
    cv2.imshow("images", np.hstack([frame, skin]))
    # if the 'q' key is pressed, stop the loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()