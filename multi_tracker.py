import cv2
from multiprocessing import Process, Queue

class MultiTracker:    
    trackers = []
    frame = None

    def initTracker(self, frame, bbox, iq, oq):
        tracker = cv2.TrackerKCF_create()
        tracker.init(frame, bbox)

        while True:
            f = iq.get()
            if f is not None:
                success, bbox = tracker.update(f)                
                oq.put((success, bbox))
            else:
                oq.put(None)
                
    def update(self, frame):
        result = []
        for tracker in self.trackers:
            id, track, iq, oq = tracker
            iq.put(frame)

        for idx, tracker in enumerate(self.trackers):
            id, track, iq, oq = tracker
            success, bbox = oq.get()
            if success:
                result.append((id, bbox))
            else:
                iq.close()
                oq.close()
                track.terminate()
                self.trackers.pop(idx)
            
        return result

    def add(self, id, frame, bbox):
        inputQueue = Queue()
        outputQueue = Queue()
        
        t = Process(target=self.initTracker, args=(frame, bbox, inputQueue, outputQueue))
        t.daemon = True
        t.start()

        self.trackers.append((            
            id,
            t,
            inputQueue,
            outputQueue            
        ))