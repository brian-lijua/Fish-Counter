import cv2
import numpy as np
import time
from imutils.video import FPS, FileVideoStream, WebcamVideoStream
from multiprocessing import Process, Queue
import multi_tracker as MT

multiTracker = MT.MultiTracker() # Initialize Mutli-Tracker

def showMe(frameQueue):
    while True:
        frame, total, fps, vw, vh, line_thick = frameQueue.get()
        if frame is not None:            
            counter_text = 'Total Fish: {}'.format(total)
            counter_text_size, counter_text_baseline = cv2.getTextSize(counter_text, cv2.FONT_HERSHEY_PLAIN, 1, 1)
            frame = cv2.rectangle(frame, (5, 0), (counter_text_size[0] + 5, 20), (0,0,255), -1)
            frame = cv2.putText(frame, counter_text, (5, 15), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)

            fps_text = 'FPS: {}'.format(round(fps.fps(), 2))
            fps_text_size, fps_text_baseline = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_PLAIN, 1, 1)    
            frame = cv2.putText(frame, fps_text, (vw - fps_text_size[0], 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)

            cv2.imshow('ttt', frame)
            cv2.waitKey(30)

def main():        

    # Initialize background substractor
    # Currently CV2 provide 2 types, "createBackgroundSubtractorKNN" and "createBackgroundSubtractorMOG2"
    # 1. "createBackgroundSubtractorKNN" 
    #   - subtraction is alot cleaner if background color more consistence
    # 2. "createBackgroundSubtractorMOG2" 
    #   - substraction based on Machine Learning (ML), more cleaner if background have too many color for KNN to work.
    #   - Result of substraction is not as complete compare to KNN
    backSub = cv2.createBackgroundSubtractorKNN(history=450, dist2Threshold=150.0, detectShadows=True)

    # cap = cv2.VideoCapture('fish4.mp4')
    #cap = FileVideoStream('fish4.mp4').start() #initialize file video reader
    cap = WebcamVideoStream(0)
    cap.start()
    
    time.sleep(1.0) # Block for 1 sec, to let "cap" buffer frames
    frame = cap.read() # Get current frame    
    vh, vw = frame.shape[:2] # Get current frame height and width, [0]: height, [1]: width
    vh = int(vh / 2)
    vw = int(vw / 2)
    videWriter = cv2.VideoWriter('out.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 30, (vw, vh)) #Initialize video writer

    # Set default center line
    line_thickness = 5 # default line thickness to 5px
    line_center = int((vw / 2)) # divide 2 with video width (vw) to get video center pixel

    fishes = [] # store fish id & tracker
    counter = 0 # fish counter

    fps = FPS().start() #Initialize FPS counter

    ttQ = Queue()
    tt = Process(target=showMe, args=(ttQ,))
    tt.daemon = True
    tt.start()

    # While cap has more frame continue
    while True:            
        frame = cv2.resize(frame, (vw, vh), cv2.INTER_AREA)
        blank_frame = frame.copy() # Create a copy of original image for filtering    
        trackers = multiTracker.update(frame)
        if len(trackers) > 0:
            for tracker in trackers:
                c, bbox = tracker # c = id
                #success, bbox = tracker.update(frame) # run tracker update based on current frame
                
                # Should tracker success = false, removed it from array to prevent future processes
                #if not success:
                    # fishes.pop(idx)
                    # continue
                
                # Calculate & Draw
                offset = 0
                xy1 = (int(bbox[0]) - offset, int(bbox[1]) - offset) # staring X and Y bounding box
                xy2 = (int(bbox[2]) + offset, int(bbox[3]) + offset) # Width and Height bounding box

                # Note:
                # OpenCV Tracker return Region of Interest (ROI) which consist
                # 1. starting point X and Y coordinate in pixel
                # 2. width and height of the bounding box in pixel 
                # OpenCV rectangle function however is draw using 2 set of coordinate in pixel
                # starting point (Top Left) and end point (Bottom Right)
                # "xy1" is the starting coordinate (Top Left), as such using
                #     ((xy1[0] + xy2[0]), (xy1[1] + xy2[1]))
                # we can calculate the 2nd set coordinate (Bottom Right)            
                # Bottom = starting X point + bounding box width        
                # Right = starting Y point + bounding box height    
                cv2.rectangle(frame, xy1, ((xy1[0] + xy2[0]), (xy1[1] + xy2[1])), (0,0,255), 1)
                cv2.putText(frame, 'id: {}'.format(c), (xy1[0], xy1[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1, cv2.LINE_AA)
                cv2.rectangle(blank_frame, xy1, ((xy1[0] + xy2[0]), (xy1[1] + xy2[1])), (255, 255, 255), -1)

        # Note:
        # OpenCV "findContours" function only work with image in gray color    
        gFrame = cv2.cvtColor(blank_frame, cv2.COLOR_RGB2GRAY) # Convert image to gray
        fMask = backSub.apply(gFrame) # Apply background seperation algorythm
        fMask = cv2.morphologyEx(fMask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8), iterations=2) # Fix deform contour
        fMask = cv2.morphologyEx(fMask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), iterations=2) # Fix deform contour    
        fMask = cv2.bitwise_and(gFrame, gFrame, mask=fMask) # combine targeted frame with mask    
        fMask = cv2.GaussianBlur(fMask, (5,5), 0) # add blur to further reduce pixel deform
        ret, thresh = cv2.threshold(fMask, 50, 255, cv2.THRESH_BINARY) # Create threshold algorythm
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # find contours

        # Loop through all found contours
        # Should any contour found and is within "center_line" tag and track it
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)        
            if x + w >= int(line_center) and x + w <= int(line_center + line_thickness):
                counter += 1
                # tracker = cv2.TrackerCSRT_create()
                # tracker.init(frame, (int(x), int(y), int(w), int(h)))
                # fishes.append((counter, tracker))

                multiTracker.add(counter, frame, (int(x), int(y), int(w), int(h)))
                # cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0,0,255), 1)

        # Draw "line_center"
        frame = cv2.rectangle(frame, (line_center, vh), (line_center + line_thickness, 0), (0,0,0), -1)
        # Calculate, Generate and Draw "Total Fish" text
        
        # Calculate, Generate and Draw "FPS" text
        fps.update()
        fps.stop()
        
        tt = (
            frame,
            counter,
            fps,
            vw,
            vh,
            line_thickness
        )
        ttQ.put(tt)

        # Display the final combine of the orignal frame including tracked item
        # cv2.imshow('frame', frame)
        # cv2.imshow('mask', fMask)
        # Display frame is being refresh every 30ms, change to 0 if manual forward required 
        key = cv2.waitKey(30)
        # should "Q" is press, stop loop
        if key == ord('q'):
            break

        # videWriter.write(frame) # Write frame to video output
        # ok, frame = cap.read()
        frame = cap.read()


if __name__ == '__main__':
    main()