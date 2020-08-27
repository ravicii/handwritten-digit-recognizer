import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train,x_test=x_train.reshape(60000,28,28,1),x_test.reshape(10000,28,28,1)
model=tf.keras.Sequential([
    tf.keras.layers.Conv2D(16,(5,5),input_shape=(28,28,1),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(8,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10,activation='softmax')
])
model.compile(tf.keras.optimizers.Adam(learning_rate=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()
dataGen=ImageDataGenerator(
    rescale=1/255.0,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    rotation_range=20
    )
dataGen.fit(x_train)
x_test=x_test/255.0
y_train=to_categorical(y_train,10)
y_test=to_categorical(y_test,10)
history=model.fit_generator(dataGen.flow(x_train,y_train,batch_size=50),epochs=10,validation_data=(x_test,y_test))
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