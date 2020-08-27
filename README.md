# handwritten-digit-recognizer
A simple CNN model that recognizes handwritten digits .
Trained on mnist dataset.
# How to use
Check the model architecture in the create_model.py file .
I have alreday included a pre-trained model so no need to create the model again.
Run predict.py to use the model to give prediction.
I use contour to detect where the digits are and then pre-process it to feed it into the cnn to get the prediction.
following packages are required
tensorflow,
cv2,
numpy
