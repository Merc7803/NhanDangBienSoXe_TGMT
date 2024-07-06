import cv2
import torch
import time
import argparse
import function.utils_rotate as utils_rotate
import function.helper as helper

# argument parser for video input
ap = argparse.ArgumentParser()
ap.add_argument('-v', '--video', required=True, help='path to input video')
args = ap.parse_args()

# Load YOLO models
yolo_LP_detect = torch.hub.load('yolov5', 'custom', path='model/LP_detector.pt', force_reload=True, source='local')
yolo_license_plate = torch.hub.load('yolov5', 'custom', path='model/LP_ocr.pt', force_reload=True, source='local')
yolo_license_plate.conf = 0.60

# Open video capture
vid = cv2.VideoCapture(args.video)
if not vid.isOpened():
    print("Error: Could not open video.")
    exit()

# Create a named window
cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
# Resize the window
cv2.resizeWindow('frame', 1200, 600)

# Process frame of the video
while True:
    ret, frame = vid.read()
    if not ret:
        break
    
    plates = yolo_LP_detect(frame, size=640)
    list_plates = plates.pandas().xyxy[0].values.tolist()
    list_read_plates = set()
    
    if len(list_plates) == 0:
        lp = helper.read_plate(yolo_license_plate, frame)
        if lp != "unknown":
            cv2.putText(frame, lp, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            list_read_plates.add(lp)
    else:
        for plate in list_plates:
            flag = 0
            x = int(plate[0])  
            y = int(plate[1])  
            w = int(plate[2] - plate[0])  
            h = int(plate[3] - plate[1])  
            crop_img = frame[y:y+h, x:x+w]
            cv2.rectangle(frame, (int(plate[0]), int(plate[1])), (int(plate[2]), int(plate[3])), color=(0, 0, 225), thickness=2)
            cv2.imwrite("crop.jpg", crop_img)
            rc_image = cv2.imread("crop.jpg")
            lp = ""
            for cc in range(2):
                for ct in range(2):
                    lp = helper.read_plate(yolo_license_plate, utils_rotate.deskew(crop_img, cc, ct))
                    if lp != "unknown":
                        list_read_plates.add(lp)
                        cv2.putText(frame, lp, (int(plate[0]), int(plate[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                        flag = 1
                        break
                if flag == 1:
                    break
    
    # Display the result
    cv2.imshow('frame', frame)
    
    # q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture
vid.release()
cv2.destroyAllWindows()
