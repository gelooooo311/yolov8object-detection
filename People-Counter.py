import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import datetime as dt
import math
from sort import *

cap = cv2.VideoCapture("../Videos/video4.mp4")  # For Video

model = YOLO("../Yolo-Weights/yolov5su.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

mask = cv2.imread("1111.png")

def POINTS(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)

# Tracking
tracker = Sort(max_age=10, min_hits=5, iou_threshold=0.5)


jaywalks = [0, 625, 1278, 648]

jaywalkCounter = []
#totalCountDown = []

while True:
    success, img = cap.read()
    imgRegion = cv2.bitwise_and(img, mask)

    imgGraphics = cv2.imread("graphics.png", cv2.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(img, imgGraphics, (730, 29))
    results = model(imgRegion, stream=True)

    detections = np.empty((0, 5))
    timestamp = dt.datetime.now().strftime("%Y/%m/%d %H:%M:%S")


    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass == 'person' and conf >= 0.5:
                 #cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1)),
                 #                   scale=1, thickness=2, offset=1)
                 #cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=5)
                 currentArray = np.array([x1, y1, x2, y2, conf])
                 detections = np.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections)

    cv2.line(img, (jaywalks[0], jaywalks[1]), (jaywalks[2], jaywalks[3]), (0, 0, 255), 5)
    #cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 0, 255), 5)

    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(img,f'{int(id)}', (max(0, x1), max(35, y1)),
                           scale=2, thickness=2, offset=10)

        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        if jaywalks[0] < cx < jaywalks[2] and jaywalks[1] - 15 < cy < jaywalks[1] + 15:
            if jaywalkCounter.count(id) == 0:
                jaywalkCounter.append(id)
                cv2.line(img, (jaywalks[0], jaywalks[1]), (jaywalks[2], jaywalks[3]), (0, 255, 0), 5)

        #if limitsDown[0] < cx < limitsDown[2] and limitsDown[1] - 15 < cy < limitsDown[1] + 15:
        #    if totalCountDown.count(id) == 0:
        #        totalCountDown.append(id)
        #        cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 255, 0), 5)
    # # cvzone.putTextRect(img, f' Count: {len(totalCount)}', (50, 50))
    cv2.putText(img,str(len(jaywalkCounter)),(1208, 112),cv2.FONT_HERSHEY_PLAIN,5,(139,195,75),7)
    cv2.putText(img, timestamp, (20, 20),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),2)
    #cv2.putText(img,str(len(totalCountDown)),(1191,345),cv2.FONT_HERSHEY_PLAIN,5,(50,50,230),7)

    cv2.imshow("FRAME", img)
    cv2.setMouseCallback('FRAME', POINTS)
    # cv2.imshow("ImageRegion", imgRegion)
    cv2.waitKey(1)
