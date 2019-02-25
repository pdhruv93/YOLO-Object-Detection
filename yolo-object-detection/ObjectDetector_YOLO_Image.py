
# coding: utf-8

# In[13]:


import numpy as np
import cv2
import time


# In[14]:


imagePath="E:\\ML_master docs\\YOLO Object Detection\\yolo-object-detection\\images\\1.jpg"
yoloPath="E:\\ML_master docs\\YOLO Object Detection\\yolo-object-detection\\yolo-coco"
# command line arguments in dict form
#image-- path to input image
#yolo --base path to YOLO directory
#confidence --minimum probability to filter weak detections
#threshold --threshold when applyong non-maxima suppression
args={'image': imagePath , 'yolo': yoloPath , 'confidence': 0.5, 'threshold': 0.3} 


# In[15]:


# load the COCO class labels our YOLO model was trained on
LABELS = open(yoloPath+"\\coco.names").read().strip().split("\n")
print("Toal classes {0}".format(len(LABELS)))


# In[16]:


# initialize a list of colors to represent each possible class label
np.random.seed(42)
#create random list of int type numbers from range 0-255. Size = len(LABELS), 3... 3 is for RGB
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),dtype="uint8")


# In[17]:


# derive the paths to the YOLO weights and model configuration
weightsPath = yoloPath+"\\yolov3.weights"
configPath = yoloPath+"\\yolov3.cfg"
# load our YOLO object detector trained on COCO dataset (80 classes)
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)


# In[18]:


# load our input image and grab its spatial dimensions
image = cv2.imread(args["image"])
print(image.shape) # height, width, channgels
(H, W) = image.shape[:2]


# In[19]:


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


# In[20]:


# construct a blob from the input image 
#why we do : preprocessing images and preparing them for classification via pre-trained deep learning models.
blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),swapRB=True, crop=False)

net.setInput(blob)

start = time.time()
#perform a forward pass of the YOLO object detector, giving us our bounding boxes and associated probabilities
layerOutputs = net.forward(ln)
end = time.time()
 
# show timing information on YOLO
print("[INFO] YOLO took {:.6f} seconds".format(end - start))


# In[21]:


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


# In[22]:


# apply non-maxima suppression to suppress weak, overlapping bounding boxes
#All that is required is that we submit our:
#bounding boxes , confidences , confidence threshold and NMS threshold
idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],args["threshold"])

#idxs now hold indexes after non maxima suppression


# In[23]:


#draw the boxes and class text on the image
if len(idxs) > 0:
    for i in idxs.flatten():
        # extract the bounding box coordinates
        (x, y) = (boxes[i][0], boxes[i][1]) #x,y--coordinate of top left corner
        (w, h) = (boxes[i][2], boxes[i][3])
 
        #pick the color
        color = [int(c) for c in COLORS[classIDs[i]]]
        
        # draw a bounding box rectangle
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)  #2 is the line thickness
        
        #prepare text
        text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
        
        #put text on image at x, y-5....a bit up then the box
        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)

# show the output image
cv2.imshow("Image", image)
cv2.waitKey(0)

