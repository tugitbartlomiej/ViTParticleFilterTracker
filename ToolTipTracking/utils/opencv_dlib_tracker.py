import cv2
import dlib

video = cv2.VideoCapture("video.mp4")
tracker = dlib.correlation_tracker()

# Inicjalizacja śledzenia
ret, frame = video.read()
bbox = cv2.selectROI("Tracking", frame, False)
tracker.start_track(frame, dlib.rectangle(*bbox))

while True:
    ret, frame = video.read()
    if not ret:
        break
    
    tracker.update(frame)
    pos = tracker.get_position()
    
    # Rysowanie pozycji
    cv2.rectangle(frame, (int(pos.left()), int(pos.top())), (int(pos.right()), int(pos.bottom())), (0,255,0), 2)
    
    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
