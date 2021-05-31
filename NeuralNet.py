import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#os.system('python -m tensorflow.tensorboard --logdir=' + 'logs/fit/2021/')
import sys
from PIL import Image
import cv2
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation, BatchNormalization, AveragePooling2D
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
import tensorflow_datasets as tfds # pip install tensorflow-datasets
import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
import tensorboard as tb
import tensorboard.program
import tensorboard.default
import logging

logger = tf.get_logger()
logger.setLevel(logging.ERROR)
def mnist_cnn_model():
   image_size = 28 # размер изображения, в пикселях
   num_channels = 1  # для черно-белых изображений
   num_classes = 11  # выходные классы
   model = Sequential() # Создание модели и слоев
   model.add(Conv2D(filters=32, kernel_size=(3,3), activation='sigmoid',
            padding='same', input_shape=(image_size, image_size, num_channels)))
   model.add(BatchNormalization(axis=1))
   model.add(MaxPooling2D(pool_size=(2, 2)))
   model.add(Dropout(0.3))
   model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='sigmoid',
            padding='same'))    
   #model.add(BatchNormalization(axis=1))
   model.add(MaxPooling2D(pool_size=(2, 2)))
   model.add(Dropout(0.3))
   model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='sigmoid',
            padding='same')) 
   #model.add(BatchNormalization(axis=1))
   model.add(MaxPooling2D(pool_size=(7, 7)))
   model.add(Dropout(0.3))
   model.add(Flatten())
   # Densely connected layers
   model.add(Dense(128, activation='sigmoid'))
   model.add(BatchNormalization(axis=1))
   # Output layer
   model.add(Dense(num_classes, activation='softmax'))
   model.compile(optimizer=Adam(), loss='categorical_crossentropy',
            metrics=['acc']) # Оптимизатор и функция потерь
   return model

def mnist_cnn_train(model):
   (train_digits, train_labels), (test_digits, test_labels) = keras.datasets.mnist.load_data()

   # Get image size
   image_size = 28
   num_channels = 1  # 1 for grayscale images
   #train_digits = normalize(train_digits)  
   # re-shape and re-scale the images data 
   img_train=cv2.imread("Obv1.jpg",0)
   img_train = cv2.resize(img_train,(image_size,image_size))
   train_labels = np.concatenate((train_labels,np.ndarray(11)))
   train_digits = np.concatenate((train_digits,[img_train]),axis=0)
   train_data = np.reshape(train_digits, (train_digits.shape[0], image_size, image_size, num_channels))
   train_data = train_data.astype('float32') / 255.0
   # encode the labels - we have 10 output classes
   # 3 -> [0 0 0 1 0 0 0 0 0 0], 5 -> [0 0 0 0 0 1 0 0 0 0]
   num_classes = 11
   train_labels_cat = keras.utils.to_categorical(train_labels, num_classes)

   # re-shape and re-scale the images validation data
   val_data = np.reshape(test_digits, (test_digits.shape[0], image_size, image_size, num_channels))
   val_data = val_data.astype('float32') / 255.0
   # encode the labels - we have 10 output classes
   val_labels_cat = keras.utils.to_categorical(test_labels, num_classes)

  # tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir,histogram_freq=1)

   print("Training the network...")
   t_start = time.time()
   EPOCHS=100

   # Start training the network
   history = model.fit(train_data[0:10000], train_labels_cat[0:10000], epochs=EPOCHS, batch_size=100,
        validation_data=(val_data, val_labels_cat))

   print("Done, dT:", time.time() - t_start)
   acc = history.history['acc']
   val_acc = history.history['val_acc']

   loss = history.history['loss']
   val_loss = history.history['val_loss']

   epochs_range = range(EPOCHS)

   plt.figure(figsize=(10,10))
   plt.subplot(1, 2, 1)
   plt.plot(epochs_range, acc, label='Точность на обучении')
   plt.plot(epochs_range, val_acc, label='Точность на валидации')
   plt.legend(loc='lower right')
   plt.title('Точность на обучающих и валидационных данных')

   plt.subplot(1, 2, 2)
   plt.plot(epochs_range, loss, label='Потери на обучении')
   plt.plot(epochs_range, val_loss, label='Потери на валидации')
   plt.legend(loc='upper right')
   plt.title('Потери на обучающих и валидационных данных')
   plt.savefig('./foo.jpg')
   plt.show()   
   return model

model = mnist_cnn_model()
mnist_cnn_train(model)
model.save('cnn_digits_28x28.h5')
def cnn_digits_predict(model, image_file):    

   image_size = 28
   img = keras.preprocessing.image.load_img(image_file, 
   target_size=(image_size, image_size), color_mode='grayscale')
   img_arr = np.expand_dims(img, axis=0)
   img_arr = 1 - img_arr/255.0
   img_arr = img_arr.reshape((1, 28, 28, 1))
   #tensorboard --logdir='C:\Users\Denis\source\repos\PythonApplication1\PythonApplication1\logs\fit\2021'                                         logdir="logs/fit/2021/"    mod_dir="model/cnn_digits_28x28.h5"
   result = model.predict_classes([img_arr]) 

   return result[0]

 

model = tf.keras.models.load_model('cnn_digits_28x28.h5')
model.summary()

print(cnn_digits_predict(model, 'tets0.png'))
print(cnn_digits_predict(model, 'tets1.png')) 
print(cnn_digits_predict(model, 'tets8.png')) 