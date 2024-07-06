import cv2
import torch
import time
import function.utils_rotate as utils_rotate
import function.helper as helper
from IPython.display import display

try:
    yolo_LP_detect = torch.hub.load('./yolov5', 'custom', path='model/LP_detector_1.pt', source='local')
    yolo_license_plate = torch.hub.load('./yolov5', 'custom', path='model/LP_ocr_1.pt', source='local')
    yolo_license_plate.conf = 0.60
except Exception as e:
    print(f"Error loading models: {e}")
    exit()

prev_frame_time = 0
new_frame_time = 0

# Test video access
vid = cv2.VideoCapture(0)
if not vid.isOpened():
    print("Error: Could not open video stream")
    exit()

while True:
    ret, frame = vid.read()
    if not ret or frame is None:
        print("Error: Failed to capture image")
        continue

    try:
        plates = yolo_LP_detect(frame, size=640)
        if plates is None:
            print("Error: YOLO detection failed")
            continue
        
        list_plates = plates.pandas().xyxy[0].values.tolist()
        list_read_plates = set()

        for plate in list_plates:
            flag = 0
            x = int(plate[0])  
            y = int(plate[1])  
            w = int(plate[2] - plate[0])  
            h = int(plate[3] - plate[1])  
            crop_img = frame[y:y+h, x:x+w]
            
            # Draw rectangle 
            cv2.rectangle(frame, (int(plate[0]), int(plate[1])), (int(plate[2]), int(plate[3])), color=(0, 0, 225), thickness=2)
            
            # Save and read cropped image for OCR processing
            cv2.imwrite("crop.jpg", crop_img)
            rc_image = cv2.imread("crop.jpg")
            
            lp = ""
            for cc in range(2):
                for ct in range(2):
                    lp = helper.read_plate(yolo_license_plate, utils_rotate.deskew(crop_img, cc, ct))
                    if lp != "unknown":
                        list_read_plates.add(lp)
                        cv2.putText(frame, lp, (int(plate[0]), int(plate[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                        flag = 1
                        break
                if flag == 1:
                    break

        # Calculate and display FPS
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        fps = int(fps)
        cv2.putText(frame, str(fps), (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)
        
        # Display the result
        cv2.imshow('frame', frame)

        # q to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print(f"Error during processing: {e}")
        continue

vid.release()
cv2.destroyAllWindows()
