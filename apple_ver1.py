import cv2
import numpy as np

cascade_path = 'cascade.xml'    #「cascade.xml」のパス指定
cap = cv2.VideoCapture(0)

cv2.namedWindow('out', cv2.WINDOW_NORMAL)
cv2.namedWindow('redout', cv2.WINDOW_NORMAL)

#モルフォロジー変換用
kernel = np.ones((20, 20), np.uint8)

def hoge(x):    #トラックバー用
    pass

#トラックバー作成
#cv2.createTrackbar('M1min_h', 'out', 0, 30, hoge)
cv2.createTrackbar('M1min_s', 'out', 0, 255, hoge)
cv2.createTrackbar('M1min_v', 'out', 0, 255, hoge)

#cv2.createTrackbar('M1max_h', 'out', 0, 30, hoge)
cv2.createTrackbar('M1max_s', 'out', 255, 255, hoge)
cv2.createTrackbar('M1max_v', 'out', 255, 255, hoge)

#cv2.createTrackbar('M2min_h', 'out', 150, 179, hoge)
cv2.createTrackbar('M2min_s', 'out', 0, 255, hoge)
cv2.createTrackbar('M2min_v', 'out', 0, 255 , hoge)

#cv2.createTrackbar('M2max_h', 'out', 150, 179, hoge)
cv2.createTrackbar('M2max_s', 'out', 255, 255, hoge)
cv2.createTrackbar('M2max_v', 'out', 255, 255, hoge)

#変数初期化
m1min_h = 0
m1min_s = 0
m1min_v = 0
m1max_h = 20
m1max_s = 255
m1max_v = 255

m2min_h = 238
m2min_s = 0
m2min_v = 0
m2max_h = 255
m2max_s = 255
m2max_v = 255

while True:
    ret, frame = cap.read() #カメラ取込み
    
    #リサイズ
    h, w = frame.shape[:2]
    height = round(h * (500 / w))
    frame = cv2.resize(frame, (500, height))
    
    dst = frame #出力用
    frame = cv2.blur(frame, (20, 20))   #ノイズ除去
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV_FULL)   #HSV変換
    
    # 赤色のHSVの値域1
    hsv_min = np.array([m1min_h, m1min_s, m1min_v])
    hsv_max = np.array([m1max_h, m1max_s, m1max_v])
    mask1 = cv2.inRange(hsv, hsv_min, hsv_max)
 
    # 赤色のHSVの値域2
    hsv_min = np.array([m2min_h, m2min_s, m2min_v])
    hsv_max = np.array([m2max_h, m2max_s, m2max_v])
    mask2 = cv2.inRange(hsv, hsv_min, hsv_max)
    red_mask = mask1 + mask2    #マスク合成

    red_img = cv2.bitwise_and(frame, frame, mask =red_mask) #赤色抽出

    red_img = cv2.morphologyEx(red_img, cv2.MORPH_CLOSE, kernel)    #モルフォロジー変換

    gray = cv2.cvtColor(red_img, cv2.COLOR_BGR2GRAY)    #グレイスケール
    
    #カスケード分類器
    cascade = cv2.CascadeClassifier(cascade_path)
    rect = cascade.detectMultiScale(gray, scaleFactor=1.07, minNeighbors=2, minSize=(1, 1))
    #rect[lefttop_x, y, width, height]
    if len(rect) > 0:
        for rect in rect:
            cv2.rectangle(dst, tuple(rect[0:2]), tuple(rect[0:2]+rect[2:4]), (0, 0, 255), thickness=3)  #矩形描画

    #出力
    cv2.imshow("out", dst)
    cv2.imshow("redout", red_img)

    key = cv2.waitKey(1)
    if key == ord('q'): #「Q」キーを押して終了
        break
    
    #トラックバー更新
    #m1min_h = cv2.getTrackbarPos('M1min_h', 'out')
    m1min_s = cv2.getTrackbarPos('M1min_s', 'out')
    m1min_v = cv2.getTrackbarPos('M1min_v', 'out')

    #m1max_h = cv2.getTrackbarPos('M1max_h', 'out')
    m1max_s = cv2.getTrackbarPos('M1max_s', 'out')
    m1max_v = cv2.getTrackbarPos('M1max_v', 'out')

    #m2min_h = cv2.getTrackbarPos('M2min_h', 'out')
    m2min_s = cv2.getTrackbarPos('M2min_s', 'out')
    m2min_v = cv2.getTrackbarPos('M2min_v', 'out')

    m2max_h = cv2.getTrackbarPos('M2max_h', 'out')
    m2max_s = cv2.getTrackbarPos('M2max_s', 'out')
    m2max_v = cv2.getTrackbarPos('M2max_v', 'out')
    
cv2.destroyAllWindows()
cap.releace()
