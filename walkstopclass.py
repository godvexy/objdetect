from ultralytics import YOLO
import cv2
import numpy as np
import pyttsx3


#initiate tts engine and define properties
engine=pyttsx3.init()
engine.setProperty('rate',100)
engine.setProperty('volume',0.7)

# Load YOLO models
model_traffic = YOLO('yolov8n.pt')
model_class = YOLO(r'C:\Users\user\Documents\sameer project\classification\new training\best_1.pt')

cap = cv2.VideoCapture(r'C:\Users\user\Desktop\sameer\walkstop.mp4')

frame_skip = 5  # Number of frames to skip between detections. Adjust based on your needs.
frame_count = 0

pred=''
while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    # Process every nth frame (where n is frame_skip)
    if frame_count % frame_skip == 0:
        results_traffic = model_traffic(frame,classes=[9],show=True,conf=0.34)
        detected_boxes = results_traffic[0].boxes.data
        detected_boxes = detected_boxes.detach().cpu().numpy()

        for box in detected_boxes:
            x1, y1, x2, y2 = map(int, box[:4])

            # Calculate the center and margins for filtering
            width = frame.shape[1]
            center_traffic = (x1 + x2) / 2
            left_margin = width * 0.25
            right_margin = width * 0.75

             # Process only if the traffic light is within the desired margins
            if left_margin < center_traffic < right_margin:
                traffic_light_img = frame[y1:y2, x1:x2]
                results_class = model_class(traffic_light_img)
                probs = results_class[0].probs.data.cpu().numpy()

                max_prob_index = np.argmax(probs)  # Index of highest probability
                max_class_name = results_class[0].names[max_prob_index]  # Class name of highest probability
                max_prob = probs[max_prob_index]  # Maximum probability value
                if pred==max_class_name or max_class_name=='ignore':
                    pass
                else:
                    engine.say(max_class_name)
                    engine.runAndWait()
                    pred=max_class_name
    frame_count += 1  # Increment frame counter
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
