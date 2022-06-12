#!/usr/bin/python3

from LitchiLocation import *
import cv2
import time 


if __name__ == "__main__":
    classes = ["q1-f5",
               "q2-f5",
               "q1-f4",
               "q1-f3",
               "q2-f4",
               "q3-f5",
               "q1-f2",
               "q2-f3",
               "q3-f1",
               "q1-f1",
               "q2-f1",
               "q2-f2",
               "q3-f4",
               "q3-f2",
               "q3-f3"]
    colors = [(0,0,0),
               (255,255,255),
               (255,0,0),
               (0,255,0),
               (0,0,255),
               (255,0,255),
               (255,255,0),
               (0,255,255),
               (192,192,192),
               (128,128,128),
               (128,0,0),
               (128,128,0),
               (0,128,0),
               (128,0,128),
               (0,128,128),
               (0,0,128)]
    a = LitchiLocation("model/litchi.param", "model/litchi.bin")
    cap = cv2.VideoCapture(r"./image/test1.avi")
    while True:
        ret,frame = cap.read()
        if not ret:
            break
        start_time = time.time()
        res = a.locate(frame)
        end_time = time.time()
        for i in range(0,len(res),6):
            label = res[i]
            cor1 = int(res[i+1])
            cor2 = int(res[i+2])
            cor3 = int(res[i+3])
            cor4 = int(res[i+4])
            confident = res[i+5]
            #cv2.putText(frame, '{0}'.format(classes[label],))
            cv2.rectangle(frame,(cor1,cor2),(cor3,cor4),colors[label],2)
        print(label, str(end_time - start_time))
        cv2.imshow("33",frame)
        cv2.waitKey(10)
    cap.realse()
    cv2.destroyAllWindows()
    
