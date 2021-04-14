
#permissions (to download data set from internet)
import ssl
ssl._create_default_https_context=ssl._create_unverified_context

#code starts
import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras import layers


#printing stuff
import matplotlib.pyplot as plt

#load a pre-defined dattaset (7 0k images of diff objects(clothes etc.)of 28*28)
fashion_mnist =keras.datasets.fashion_mnist  #(mnst contains all the fashion stuff to be loaded)
'''when u print train images there will be a 28*28 numpy arrat with 1's and 0's...0=black & 1=white'''

#pull out data from datasett
(train_images,train_labels),(test_images , test_labels) =fashion_mnist.load_data()

#show datasets
#print(train_labels[0])
#print(train_images[0])                                                                                                              
'''plt.imshow(train_images[0],cmap='gray',vmin=0,vmax=255)
plt.show()'''

#define our neural network structure
'''in a sequential neural network containg vertical columns and then they go in  row'''
model=keras.Sequential()

model.add(layers.Flatten(input_shape=(28,28)))
#input is a 28*28 image("flatten" flattens the 28*28 into a single 784*1 input layer)

model.add(layers.Dense(units=128, activation=tf.nn.relu))
#hidden layer is 128 deep. relu returns the value, or 0(works good enough much faster)

model.add(layers.Dense(units=10,activation=tf.nn.softmax))
#output is 0-10(depending on cloth).return maximum


#complete our model 
model.compile(optimizer=tf.optimizers.Adam(),loss='sparse_categorical_crossentropy',metrics=['accuracy'])

#train our model using our testing data
model.fit(train_images,train_labels,epochs=5)

#test our model,usingour testing data
test_loss=model.evaluate(test_images,test_labels)

#output the images(from keras)
plt.imshow(test_images[1],cmap='gray',vmin=0,vmax=255)
plt.show()

#make predictions 
predictions=model.predict(test_images)

#print(predictions[1]).....this is to get the list of probabilities

#print out predicting
print(list(predictions[1]).index(max(predictions[1])))

