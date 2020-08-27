# handwritten-digit-recognizer
A simple CNN model that recognizes handwritten digits .\n
Trained on mnist dataset.\n
# How to use
Check the model architecture in the create_model.py file .\n
I have alreday included a pre-trained model so no need to create the model again.\n
Run predict.py to use the model to give prediction.\n
I use contour to detect where the digits are and then pre-process it to feed it into the cnn to get the prediction.\n
following packages are required\n
tensorflow\n
cv2\n
numpy
