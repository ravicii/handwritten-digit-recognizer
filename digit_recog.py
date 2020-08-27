import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train=x_train.reshape(60000,28,28,1)
x_test=x_test.reshape(10000,28,28,1)
#remove the comments below if you are running this for the first time to create the model
'''
model=tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(8,(3,3),input_shape=(28,28,1),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(16,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10,activation='softmax')
])
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
dataGen=tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1/255.0,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    shear_range=0.1,
    rotation_range=15
    )
dataGen.fit(x_train)
x_test=x_test/255.0
y_train=tf.keras.utils.to_categorical(y_train,10)
y_test=tf.keras.utils.to_categorical(y_test,10)
history=model.fit_generator(dataGen.flow(x_train,y_train,batch_size=50),epochs=5,validation_data=(x_test,y_test))
plt.figure(1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.xlabel('EPOCHS')
plt.ylabel('Accuracy')
plt.legend(['Training Accuracy','Test Accuracy'])
plt.figure(2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('EPOCHS')
plt.ylabel('Loss')
plt.legend(['Training Loss','Test Loss'])
plt.show()
model.save('digit-recog-model.h5')
'''
#load an image file from your computer
model=tf.keras.models.load_model('digit-recog-model.h5')
img=cv2.imread('2.jpg')
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
grayinvt=cv2.threshold(gray,150,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
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
grayinvt=grayinvt.reshape(1,28,28,1)
predi=model.predict(grayinvt)
print(np.argmax(predi[0]))
cv2.imshow('image',img)
cv2.waitKey(0)
#realtime prediction with webcam
'''
model=tf.keras.models.load_model('digit-recog-model.h5')
cap=cv2.VideoCapture(0)
while (True):
    ret,frame=cap.read()
    cv2.imshow('capture',frame)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    grayinvt=cv2.threshold(gray,150,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
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
    grayinvt=grayinvt.reshape(1,28,28,1)
    predi=model.predict(grayinvt)
    print(np.argmax(predi[0]))
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
'''
'''
#check a prediction from the test dataset
#change the index to change the image from test dataset
model=tf.keras.models.load_model('digit-recog-model.h5')
predi=model.predict(x_test)
print(np.argmax(predi[11]))#change the index to change the image
plt.imshow(x_test[11],cmap=plt.cm.binary)#change the index to change the image
plt.show()
'''