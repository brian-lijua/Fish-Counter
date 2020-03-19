import logging, os, uuid
from queue import Queue
from threading import Thread
from imutils.video import FileVideoStream, FPS
import cv2
import time
import numpy as np
import multi_tracker as MultiTracker

log_format = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s')
log_stream = logging.StreamHandler()
log_stream.setFormatter(log_format)
logger = logging.getLogger(__name__)
logger.addHandler(log_stream)
logger.setLevel(logging.INFO)

FRAME_SCALE = 50

fps = FPS()

video_output_name: str = uuid.uuid4

def write_to_video(dq, videoWidth, videoHeight):
    videoWriter = cv2.VideoWriter('{}.avi'.format(video_output_name), cv2.VideoWriter_fourcc('M','J','P','G'), 30, (videoWidth, videoHeight))
    while True:
        (total, frame) = dq.get()        
        h, w = frame.shape[:2]
        c = int(w / 2)
        
        frame = cv2.rectangle(frame, (c, 0), (int(c + 3) , h), (0,0,0), -1) # center line
        frame = cv2.rectangle(frame, (38, 15), (300, 55), (0,0,255), -1)
        frame = cv2.putText(frame, 'Total: {}'.format(total), (50, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2)
        
        if frame is not None:            
            videoWriter.write(frame)
        cv2.waitKey(30)
        dq.task_done()

def run(video_path):

    if not os.path.isfile(video_path):
        logger.error('{} path is invalid'.format(video_path))

    logger.info('start processing: {}'.format(video_path))

    start_process_time = time.time()

    counter = 0
    cap = FileVideoStream(video_path).start()
    time.sleep(1.0)

    frame = cap.read()
    vh, vw = frame.shape[:2]

    dq = Queue()    
    displayInput = Thread(target=write_to_video, args=(dq, vw, vh, ))
    displayInput.daemon = True
    displayInput.start()

    mt = MultiTracker.MultiTracker()
    mt.setThreshold(abs(0.8 * (vw / 2)))

    subtractor = cv2.bgsegm.createBackgroundSubtractorGSOC(beta=0.03)
    
    fps.start()

    while cap.more():
        # frame = cv2.flip(frame, 1)
        small_frame = cv2.resize(frame, dsize=None, dst=None , fx=(50 / 100), fy=(50 / 100), interpolation=cv2.INTER_LANCZOS4)        
        filtered_frame = small_frame.copy()

        # convert to gray
        grayFrame = cv2.cvtColor(filtered_frame, cv2.COLOR_BGR2GRAY)
        grayFrame = cv2.GaussianBlur(grayFrame, (5, 5), 0)
        
        mask = grayFrame.copy()
        mask = subtractor.apply(mask)

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
            if cX >= int(vw / 4) and cX <= int((vw / 4) + 28):
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

    dq.join()    
    cap.stop()

    end_process_time = time.time()
    logger.info('Completed - duration: {}, video: {}'.format(int(end_process_time - start_process_time), video_path))

if __name__ == '__main__':
    run('fish_counting_ai.mp4')