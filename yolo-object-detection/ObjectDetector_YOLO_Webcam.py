
# coding: utf-8

# In[4]:


from imutils.video import VideoStream
from imutils.video import FPS
import time
import numpy as np
import cv2
import imutils


# In[5]:


yoloPath="E:\\ML_master docs\\YOLO Object Detection\\yolo-object-detection\\yolo-coco"
#yolo --base path to YOLO directory
#confidence --minimum probability to filter weak detections
#threshold --threshold when applyong non-maxima suppression
args={'yolo': yoloPath , 'confidence': 0.5, 'threshold': 0.3}


# In[6]:


# load the COCO class labels our YOLO model was trained on
LABELS = open(yoloPath+"\\coco.names").read().strip().split("\n")
print("Toal classes {0}".format(len(LABELS)))


# In[7]:


# initialize a list of colors to represent each possible class label
np.random.seed(42)
#create random list of int type numbers from range 0-255. Size = len(LABELS), 3... 3 is for RGB
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),dtype="uint8")


# In[8]:


# derive the paths to the YOLO weights and model configuration
weightsPath = yoloPath+"\\yolov3.weights"
configPath = yoloPath+"\\yolov3.cfg"
# load our YOLO object detector trained on COCO dataset (80 classes)
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)


# In[9]:


#Generally in a sequential CNN network there will be only one output layer at the end. 
#In the YOLO v3 architecture we are using there are multiple output layers giving out predictions.
ln_all = net.getLayerNames()
#print(ln)

ln=[]
# determine only the *output* layer names that we need from YOLO
for i in ln_all:
    if "yolo" in i:
        ln.append(i)
        
print(ln)


# In[10]:


# initialize the video stream, allow the cammera sensor to warmup and initialize the FPS counter
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()


# In[11]:


try:
    # loop over frames from the video file stream
    while True:
        
        # grab the frame from stream
        frame = vs.read()
        #resize it to have a maximum width of 400 pixels
        frame = imutils.resize(frame, width=400)

        (H, W) = frame.shape[:2]

        # construct a blob from the input frame
        #why we do : preprocessing images and preparing them for classification via pre-trained deep learning models.
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),swapRB=True, crop=False)

        net.setInput(blob)

        #perform a forward pass of the YOLO object detector, giving us our bounding boxes and associated probabilities
        layerOutputs = net.forward(ln)

        #yolo has now processed our image and got the data. we now just need to fetch that data
        # initialize empty lists
        boxes = []
        confidences = []
        classIDs = []

        #Let’s begin populating these lists with data from our YOLO layerOutputs
        # loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability) of the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                # filter out weak predictions by ensuring the detected probability is greater than the minimum probability
                if confidence > args["confidence"]:
                    # scale the bounding box coordinates back relative to the size of the image
                    #YOLO returns the center (x, y)-coordinates of the bounding box followed by the box width and height
                    box = detection[0:4] * np.array([W, H, W, H])

                    #astype("int") witll convert box values to int values
                    (centerX, centerY, width, height) = box.astype("int")

                    # use the center (x, y)-coordinates to derive the top-left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # update our list of bounding box coordinates, confidences and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        # apply non-maxima suppression to suppress weak, overlapping bounding boxes
        #All that is required is that we submit our:
        #bounding boxes , confidences , confidence threshold and NMS threshold
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],args["threshold"])

        #idxs now hold indexes after non maxima suppression


        #draw the boxes and class text on the frame
        if len(idxs) > 0:
            for i in idxs.flatten():
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1]) #x,y--coordinate of top left corner
                (w, h) = (boxes[i][2], boxes[i][3])

                #pick the color
                color = [int(c) for c in COLORS[classIDs[i]]]

                # draw a bounding box rectangle
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)  #2 is the line thickness

                #prepare text
                text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])

                #put text on image at x, y-5....a bit up then the box
                cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)

        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
 
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
 
        # update the FPS counter
        fps.update()

except:
    pass

# release the file pointers
# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
 
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()

