import cv2 as cv 
import numpy as np

# Create the video capture
capture = cv.VideoCapture(0)
classes = ["background", "person", "bicycle", "car", "motorcycle",
  "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
  "unknown", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
  "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "unknown", "backpack",
  "umbrella", "unknown", "unknown", "handbag", "tie", "suitcase", "frisbee", "skis",
  "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
  "surfboard", "tennis racket", "bottle", "unknown", "wine glass", "cup", "fork", "knife",
  "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
  "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "unknown", "dining table",
  "unknown", "unknown", "toilet", "unknown", "tv", "laptop", "mouse", "remote", "keyboard",
  "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "unknown",
  "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush" ]
 
# Colors we will use for the object labels
colors = np.random.uniform(0, 255, size=(len(classes), 3))

pb  = 'frozen_inference_graph.pb'
pbt = 'ssd_inception_v2_coco_2017_11_17.pbtxt'
 
# Read the neural network
cvNet = cv.dnn.readNetFromTensorflow(pb,pbt)  

# Until we press the 'd' character the video stream will keep returning frames
while True:
    isTrue, frame = capture.read()
    rows = frame.shape[0]
    cols = frame.shape[1]
    cvNet.setInput(cv.dnn.blobFromImage(frame, size=(300, 300), swapRB=True, crop=False))

    cvOut = cvNet.forward()
    for detection in cvOut[0,0,:,:]:
        score = float(detection[2])
        if score > 0.3:
            idx = int(detection[1])   # prediction class index. 
            # If you want all classes to be labeled instead of just forks, spoons, and knives, 
            # remove this line below (i.e. remove line 65)
            left = detection[3] * cols
            top = detection[4] * rows
            right = detection[5] * cols
            bottom = detection[6] * rows
            cv.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (23, 230, 210), thickness=2)
                
            # draw the prediction on the frame
            label = "{}: {:.2f}%".format(classes[idx],score * 100)
            y = top - 15 if top - 15 > 15 else top + 15
            cv.putText(frame, label, (int(left), int(y)),cv.FONT_HERSHEY_SIMPLEX, 0.5, colors[idx], 2)
    cv.imshow('my webcam', frame)
    if cv.waitKey(20) & 0xFF==ord('d'):
        break

capture.release()
cv.destroyAllWindows()
