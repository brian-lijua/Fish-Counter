from queue import Queue
from threading import Thread
from imutils.video import FileVideoStream, FPS
import cv2
import time
import numpy as np

import multi_tracker as MultiTracker

FRAME_SCALE = 50

fps = FPS()

def display(dq):
    while True:
        frame = dq.get()
        h, w = frame.shape[:2]        
        c = int(w / 2)

        frame = cv2.rectangle(frame, (c, 0), (int(c + 3) , h), (0,0,0), -1)

        if frame is not None:
            cv2.imshow('Win', frame)
        cv2.waitKey(30)
        dq.task_done()

def main():
    counter = 0
    dq = Queue()    
    displayInput = Thread(target=display, args=(dq,))    
    displayInput.daemon = True
    displayInput.start()

    mt = MultiTracker.MultiTracker()

    cap = FileVideoStream('fish7.mp4').start()
    time.sleep(1.0)

    frame = cap.read()
    vh, vw = frame.shape[:2]

    subtractor = cv2.createBackgroundSubtractorKNN(history=50, dist2Threshold=120, detectShadows=False)
    # subtractor = cv2.createBackgroundSubtractorMOG2(history=120, varThreshold=100, detectShadows=False)

    fps.start()

    while cap.more():
        # frame = cv2.flip(frame, 1)
        small_frame = cv2.resize(frame, dsize=None, dst=None , fx=(50 / 100), fy=(50 / 100), interpolation=cv2.INTER_LANCZOS4)
        filtered_frame = small_frame.copy()
        trackers = mt.update(small_frame)
        if len(trackers) > 0:
            for tracker in trackers:
                c, bbox = tracker

                offset = 0
                xy1 = (int(bbox[0]) - offset, int(bbox[1]) - offset)
                xy2 = (int(bbox[2]) + offset, int(bbox[3]) + offset)

                cv2.rectangle(small_frame, xy1, ((xy1[0] + xy2[0]), (xy1[1] + xy2[1])), (0,0,255), 1)
                cv2.putText(small_frame, 'id: {}'.format(c), (xy1[0], xy1[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1, cv2.LINE_AA)
                cv2.rectangle(filtered_frame, xy1, ((xy1[0] + xy2[0]), (xy1[1] + xy2[1])), (255, 255, 255), -1)

        grayFrame = cv2.cvtColor(filtered_frame, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('gray', grayFrame)
        grayFrame = cv2.GaussianBlur(grayFrame, (5, 5), 0)

        mask = subtractor.apply(grayFrame)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8), iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), iterations=2)
        mask = cv2.bitwise_and(grayFrame, grayFrame, mask=mask)
        # H = cv2.Sobel(mask, cv2.CV_8U, 0, 1)
        # V = cv2.Sobel(mask, cv2.CV_8U, 1, 0)
        # mask = H + V
        cv2.imshow('mask', mask)
        cv2.waitKey(30)
        _, threshold = cv2.threshold(mask, 25, 150, 0)
        contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)       
            
            if x >= int(vw / 4) and x <= int((vw / 4) + 3):
                counter += 1
                mt.add(counter, small_frame, (int(x), int(y), int(w), int(h)))
            else:
                small_frame = cv2.rectangle(small_frame, (x, y), (x + w, y + h), (0,0,255), cv2.FILLED)

        small_frame = cv2.resize(small_frame, dsize=None, dst=None, fx=(2), fy=(2), interpolation=cv2.INTER_LANCZOS4)

        fps.stop()
        fps.update()
        # print(fps.fps())
        cv2.imshow('t', small_frame)
        cv2.waitKey(0)

        dq.put(small_frame)
        frame = cap.read()

    print('Main thread done, waiting all thread queue to be done')
    dq.join()
    print('ENDING')
    cap.stop()

if __name__ == '__main__':
    main()