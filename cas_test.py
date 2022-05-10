import cv2
import numpy as np

cascade_path = 'cascade.xml'
img_out_path = '出力先パス'
i = 0
cap = cv2.VideoCapture(2)

while True:
    ret, frame = cap.read()
    dst = frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cascade = cv2.CascadeClassifier(cascade_path)
    rect = cascade.detectMultiScale(gray, scaleFactor=1.07, minNeighbors=2, minSize=(1, 1))
    #rect[lefttop_x, y, width, height]
    if len(rect) > 0:
        for rect in rect:
            cv2.imwrite(str(img_out_path) + 'img' + str(i) + '.png', frame[rect[1] : rect[1] + rect[3], rect[0] : rect[0] + rect[2]])
            cv2.imshow('apple', frame[rect[1] : rect[1] + rect[3], rect[0] : rect[0] + rect[2]])
            cv2.waitKey(1)
            print(rect)
            print("---------------------------------")
            cv2.rectangle(dst, tuple(rect[0:2]), tuple(rect[0:2]+rect[2:4]), (0, 0, 255), thickness=3)
            i += 1
    cv2.imshow("out", dst)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cv2.destroyAllWindows()
cap.releace
