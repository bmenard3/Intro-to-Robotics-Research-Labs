import cv2
import numpy as np
from PIL import Image

cap = cv2.VideoCapture(0)
params = cv2.SimpleBlobDetector_Params()


while(1):
    _, frame = cap.read()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_limit = np.array([100,75,75])
    upper_limit = np.array([120,255,255])

    mask = cv2.inRange(hsv, lower_limit, upper_limit)

    masked_frame = cv2.bitwise_and(frame,frame, mask=mask)
    inv_mask = cv2.bitwise_not(mask)

    mask_img = Image.fromarray(mask)
    bbox = mask_img.getbbox()

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        xc = int((x1 + x2) / 2)
        yc = int((y1 + y2) / 2)
        frame = cv2.circle(frame, (xc, yc), 10, (0,0,255), 3)

    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    #cv2.imshow('res', res)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()

