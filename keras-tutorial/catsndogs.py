import warnings
warnings.simplefilter('ignore') 
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.simplefilter('ignore', category=FutureWarning)

import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import pickle
import time

NAME = "Cats-vs-dogs-cnn-64x2-{}".format(int(time.time()))

tensorboard = TensorBoard(log_dir=f'/tmp/logs/{NAME}')

pickle_in = open("/storage/PetImages/X.pickle", "rb")
X = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open("/storage/PetImages/y.pickle", "rb")
y = pickle.load(pickle_in)
pickle_in.close()

print('-------- Shape of dataset ----------')
print(X.shape, y.shape)
print('------------------------------------')

print('Normalizing features...')
X = X / 255.0

print('Creating CNN 2D Model...')


model = Sequential()

model.add(Conv2D(64, (3, 3), activation='relu', input_shape = X.shape[1:]))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten()) # Flattens 3D feature to 1D features
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))


model.compile(loss="binary_crossentropy", 
             optimizer="adam",
             metrics=['accuracy'])


history = model.fit(X, y, batch_size=32, epochs=30, validation_split=0.3, callbacks=[tensorboard])



