import pytesseract
import cv2
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

img = cv2.imread('test3.jpg')
# rgb for pytesseract
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# char detector
# dims
himg, wimg, _ = img.shape
boxes = pytesseract.image_to_boxes(img)
txt = pytesseract.image_to_string(img)

print(txt)
for box in boxes.splitlines():
    dims = box.split(' ')
    x, y, w, h = int(dims[1]), int(dims[2]), int(dims[3]), int(dims[4])

    cv2.rectangle(img, (x,himg-y), (w,himg-h), (255,0,0), 1)
    cv2.putText(img, dims[0], (x, himg-y+25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)



cv2.imshow('Result', img)
cv2.waitKey(0)

#write to file
f = open('text', 'w')
f.writelines(txt)
f.close()
print('bye!')
