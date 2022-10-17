### EXP NO: 03

### .

# <p align = "center"> Convolutional Deep Neural Network for Digit Classification </p>

 

## AIM

To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images.

## Problem Statement and Dataset
The MNIST dataset is a collection of handwritten digits. The task is to classify a given image of a handwritten digit into one of 10 classes representing integer values from 0 to 9, inclusively. The dataset has a collection of 60,000 handwrittend digits of size 28 X 28. Here we build a convolutional neural network model that is able to classify to it's appropriate numerical value.

![image](https://user-images.githubusercontent.com/75235293/190975763-7d3b7c0f-9458-41e9-a35c-aa063c4977da.png)


## <br><br><br><br><br>Neural Network Model
![Screenshot (423)](https://user-images.githubusercontent.com/75243072/191222827-1d87adfd-172e-41c9-af5f-6c6cfdd7d0ac.png)



## DESIGN STEPS
### STEP-1:
Import tensorflow and preprocessing libraries
### STEP 2:
Download and load the dataset
### STEP 3:
Scale the dataset between it's min and max values
### STEP 4:
Using one hot encode, encode the categorical values
### STEP-5:
Split the data into train and test
### STEP-6:
Build the convolutional neural network model
### STEP-7:
Train the model with the training data
### STEP-8:
Plot the performance plot
### STEP-9:
Evaluate the model with the testing data
### STEP-10:
Fit the model and predict the single input

## PROGRAM
```python
# Developed By:Navneeth Krishnan B
# Register Number:212219040094
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.preprocessing import image
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train.shape
X_test.shape
single_image= X_train[0]
single_image.shape
plt.imshow(single_image,cmap='gray')
y_train.shape
X_train.min()
X_train.max()
X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0
X_train_scaled.min()
X_train_scaled.max()
y_train[0]
y_train_onehot = utils.to_categorical(y_train,10)
y_test_onehot = utils.to_categorical(y_test,10)
type(y_train_onehot)
y_train_onehot.shape
single_image = X_train[500]
plt.imshow(single_image,cmap='gray')
y_train_onehot[500]
X_train_scaled = X_train_scaled.reshape(-1,28,28,1)
X_test_scaled = X_test_scaled.reshape(-1,28,28,1)
model = keras.Sequential()
model.add(layers.Input(shape=(28,28,1))) 
model.add(layers.Conv2D(filters=32,kernel_size=(3,3),padding="same",activation='relu'))
model.add(layers.AvgPool2D (pool_size=(2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(32,activation='tanh')) 
model.add(layers.Dense(10, activation ='softmax'))
model.summary()
model.compile(optimizer='Adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(X_train_scaled ,y_train_onehot, epochs=5,
          batch_size=64, 
          validation_data=(X_test_scaled,y_test_onehot))   
metrics = pd.DataFrame(model.history.history)  
metrics.head()
metrics[['accuracy','val_accuracy']].plot()
metrics.plot()
metrics[['loss','val_loss']].plot()
x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)
print(confusion_matrix(y_test,x_test_predictions))
img = image.load_img('8.png')
type(img)
img_tensor = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(img_tensor,(28,28))
img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled = img_28_gray.numpy()/255.0
x_single_prediction = np.argmax(
    model.predict(img_28_gray_scaled.reshape(1,28,28,1)),
     axis=1)
print(x_single_prediction)
plt.imshow(img_28_gray_scaled.reshape(28,28),cmap='gray')
```


## <br>OUTPUT

### TrainingLoss, Validation Loss Vs Iteration Plot


![2022-09-23](https://user-images.githubusercontent.com/75235477/191910863-26d828eb-376f-499a-8a18-be3fc67c3c0f.png)


### Classification Report

![2022-09-23 (1)](https://user-images.githubusercontent.com/75235477/191910876-c4f4e5bd-11c7-4cc7-ae77-a38ff9c616ba.png)


### Confusion Matrix


![con](https://user-images.githubusercontent.com/75235477/191910891-5c33d041-141e-4257-8278-37bd0602e6b3.png)



### New Sample Data Prediction


![2022-09-23 (2)](https://user-images.githubusercontent.com/75235477/191910900-484d7f9d-9a10-460f-a188-b0135be30b28.png)



## RESULT
Thus,a convolutional deep neural network for digit classification is developed and the response for scanned handwritten images is verified.


