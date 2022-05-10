import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    dst = frame

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    hsv_min = np.array([0,64,0])
    hsv_max = np.array([30,255,255])
    red_mask1 = cv2.inRange(hsv, hsv_min, hsv_max)

    hsv_min = np.array([150,64,0])
    hsv_max = np.array([179,255,255])
    red_mask2 = cv2.inRange(hsv, hsv_min, hsv_max)

    # 赤色領域のマスク（255：赤色、0：赤色以外）    
    mask = red_mask1 + red_mask2

    # マスキング処理
    masked_img = cv2.bitwise_and(frame, frame, mask=mask)

    gray = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)

    cascade = cv2.CascadeClassifier('cascade.xml')
    rect = cascade.detectMultiScale(gray, scaleFactor=1.07, minNeighbors=2, minSize=(1, 1))

    if len(rect) > 0:
        for rect in rect:
            cv2.rectangle(dst, tuple(rect[0:2]), tuple(rect[0:2]+rect[2:4]), (0, 0, 255), thickness=3)

    cv2.imshow("out", dst)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cv2.destroyAllWindows()
cap.releace