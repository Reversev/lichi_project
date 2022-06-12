#!/usr/bin/python3

from LitchiLocation import *
import cv2
import time 


if __name__ == "__main__":
    
    model = LitchiLocation("model/litchi.param", "model/litchi.bin")
    cap = cv2.VideoCapture(0)
    while True:
        ret,frame = cap.read()
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
            if confident < 0.7:
                continue
            cv2.rectangle(frame,(cor1,cor2),(cor3,cor4),(0,255,0),2)
        print(str(end_time - start_time))
        cv2.imshow("33",frame)
        cv2.waitKey(10)
    cap.realse()
    cv2.destroyAllWindows()
    
