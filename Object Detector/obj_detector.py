import cv2

classNames = []
img = cv2.imread('leo.jpg')
# Files
classFile = 'coco.names'
configFile = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsFile = 'frozen_inference_graph.pb'

# fill the array
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')


# setting net
net = cv2.dnn_DetectionModel(weightsFile, configFile)
net.setInputSize(320, 320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# send img to model
classIds, confs, bbox = net.detect(img, confThreshold=0.5)

for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
    cv2.rectangle(img, box, color=(255,0,0), thickness=3)
    x = round(confidence,2)
    cv2.putText(img, f"{classNames[classId-1]} conf: {x}", (box[0]+10, box[1]+30), cv2.FONT_HERSHEY_COMPLEX, fontScale=0.75, color=(0,0,0))

# res
cv2.imshow('Output', img)
cv2.waitKey(0)
