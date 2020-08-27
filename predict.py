import cv2
import numpy as np
from tensorflow.keras.models import load_model
img=cv2.imread('test.png')
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
blur=cv2.GaussianBlur(gray,(5,5),0)
edge=cv2.Canny(blur,50,150)
contours,h=cv2.findContours(edge,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
model=load_model('digit-recog-model.h5')
key={0:'0',1:'1',2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9'}
for cnt in contours:
    area=cv2.contourArea(cnt)
    if area>100.0:
        x,y,w,h=cv2.boundingRect(cnt)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)
        grayinvt=cv2.threshold(gray[y:y+h,x:x+w],0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
        dimx,dimy=grayinvt.shape[0],grayinvt.shape[1]
        if dimx>dimy:
            r=dimy/float(dimx)
            grayinvt=cv2.resize(grayinvt,(28,int(r*28)))
        else:
            r=dimx/float(dimy)
            grayinvt=cv2.resize(grayinvt,(int(r*28),28))
        dimx,dimy=grayinvt.shape[0],grayinvt.shape[1]
        dx,dy=int((28-dimx)/2),int((28-dimy)/2)
        grayinvt=cv2.copyMakeBorder(grayinvt,top=dy,bottom=dy,
			left=dx, right=dx, borderType=cv2.BORDER_CONSTANT,
			value=(0, 0, 0))
        grayinvt=cv2.resize(grayinvt,(28,28))
        cv2.imshow('character',grayinvt)
        cv2.waitKey(0)
        grayinvt=grayinvt/255.0
        grayinvt=grayinvt.reshape((1,28,28,1))
        pred=model.predict(grayinvt)[0]
        ind=np.argmax(pred)
        cv2.putText(img,key[ind],(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
cv2.imshow('Result',img)
cv2.waitKey(0)