from queue import Queue
from threading import Thread
from imutils.video import FileVideoStream, FPS
import cv2
import time
import numpy as np

import multi_tracker as MultiTracker

FRAME_SCALE = 50

fps = FPS()

def display(dq, videoWidth, videoHeight):
    videWriter = cv2.VideoWriter('out.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 30, (videoWidth, videoHeight))
    while True:
        (total, frame) = dq.get()        
        h, w = frame.shape[:2]        
        c = int(w / 2)
        
        frame = cv2.rectangle(frame, (c, 0), (int(c + 3) , h), (0,0,0), -1) # center line
        frame = cv2.rectangle(frame, (38, 15), (300, 55), (0,0,255), -1)
        frame = cv2.putText(frame, 'Total: {}'.format(total), (50, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2)
        
        if frame is not None:
            cv2.imshow('Win', frame)
            videWriter.write(frame)
        cv2.waitKey(30)
        dq.task_done()

def main():
    counter = 0    

    cap = FileVideoStream('fish_counting_ai.mp4').start()
    time.sleep(1.0)

    frame = cap.read()
    vh, vw = frame.shape[:2]

    dq = Queue()    
    displayInput = Thread(target=display, args=(dq, vw, vh, ))    
    displayInput.daemon = True
    displayInput.start()

    mt = MultiTracker.MultiTracker()
    mt.setThreshold(abs(0.8 * (vw / 2)))

    subtractor = cv2.bgsegm.createBackgroundSubtractorGSOC(beta=0.03)
    # subtractor = cv2.createBackgroundSubtractorKNN(history=120, dist2Threshold=60, detectShadows=False)
    # subtractor = cv2.createBackgroundSubtractorMOG2(history=120, varThreshold=200, detectShadows=False)    

    fps.start()

    while cap.more():
        # frame = cv2.flip(frame, 1)
        small_frame = cv2.resize(frame, dsize=None, dst=None , fx=(50 / 100), fy=(50 / 100), interpolation=cv2.INTER_LANCZOS4)        
        filtered_frame = small_frame.copy()

        # convert to gray
        grayFrame = cv2.cvtColor(filtered_frame, cv2.COLOR_BGR2GRAY)
        grayFrame = cv2.GaussianBlur(grayFrame, (5, 5), 0)
        
        mask = grayFrame.copy()

        cv2.imshow('sobel', mask)
        # cv2.waitKey(30)

        mask = subtractor.apply(grayFrame)

        trackers = mt.update(small_frame)
        if len(trackers) > 0:
            for tracker in trackers:
                c, bbox = tracker

                offset = 0
                xy1 = (int(bbox[0]) - offset, int(bbox[1]) - offset)
                xy2 = (int(bbox[2]) + offset, int(bbox[3]) + offset)

                cv2.rectangle(small_frame, xy1, ((xy1[0] + xy2[0]), (xy1[1] + xy2[1])), (0,0,255), 1)
                cv2.putText(small_frame, 'id: {}'.format(c), (xy1[0], xy1[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1, cv2.LINE_AA)

                mXY1 = (int(xy1[0] - 15), int(xy1[1] - 15))
                mWH = (int(mXY1[0] + xy2[0] + 15), int((xy1[1] + xy2[1]) + 15) )
                mask = cv2.rectangle(mask, mXY1, mWH, (0, 0, 0), -1)
                
        cv2.imshow('mask', mask)
        cv2.waitKey(30)
        
        _, threshold = cv2.threshold(mask, 50, 100, 0)
        contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        prevContour = None
        contourThreshold = 5
        for contour in contours:
            M = cv2.moments(contour)

            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = (0, 0)
            
            cv2.circle(small_frame, (cX, cY), 2, (0, 0, 255), -1)
            if cX >= int(vw / 4) and cX <= int((vw / 4) + 30):
                if prevContour == None or abs(cX - prevContour[0]) > contourThreshold:
                    counter += 1
                    x, y, w, h = cv2.boundingRect(contour)
                    mt.add(counter, small_frame, (int(x), int(y), int(w), int(h)))
                    prevContour = (cX, cY)


        small_frame = cv2.resize(small_frame, dsize=None, dst=None, fx=(2), fy=(2), interpolation=cv2.INTER_LANCZOS4)

        fps.stop()
        fps.update()
        # print(fps.fps())
        
        dq.put((counter, small_frame))
        frame = cap.read()

    print('Main thread done, waiting all thread queue to be done')
    dq.join()
    print('ENDING')
    cap.stop()

if __name__ == '__main__':
    main()