#!/usr/bin/env python
# coding: utf-8

# <img align="left" src="https://lever-client-logos.s3.amazonaws.com/864372b1-534c-480e-acd5-9711f850815c-1524247202159.png" width=200>
# <br></br>
# 
# # Neural Network Framework (Keras)
# 
# ## *Data Science Unit 4 Sprint 2 Assignmnet 3*
# 
# ## Use the Keras Library to build a Multi-Layer Perceptron Model on the Boston Housing dataset
# 
# - The Boston Housing dataset comes with the Keras library so use Keras to import it into your notebook. 
# - Normalize the data (all features should have roughly the same scale)
# - Import the type of model and layers that you will need from Keras.
# - Instantiate a model object and use `model.add()` to add layers to your model
# - Since this is a regression model you will have a single output node in the final layer.
# - Use activation functions that are appropriate for this task
# - Compile your model
# - Fit your model and report its accuracy in terms of Mean Squared Error
# - Use the history object that is returned from model.fit to make graphs of the model's loss or train/validation accuracies by epoch. 
# - Run this same data through a linear regression model. Which achieves higher accuracy?
# - Do a little bit of feature engineering and see how that affects your neural network model. (you will need to change your model to accept more inputs)
# - After feature engineering, which model sees a greater accuracy boost due to the new features?

# In[1]:


from tensorflow.keras.datasets import boston_housing
import numpy as np


# In[11]:


(X_train, y_train), (X_test, y_test) = boston_housing.load_data()


# In[12]:


print("Before normalization")
print('Shape of X_train data:', X_train.shape)
print('Shape of X_test data:', X_test.shape)
print("Sample of X_train", X_train[0])
print("Sample of X_test", X_test[0])


# In[13]:


# Using the standard scaler
from sklearn.preprocessing import Normalizer
scaler = Normalizer()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[14]:


print("After normalization:")
print("Sample of X_train:", X_train[0])
print("Sample of y_train:", y_train[0])

print("Sample of X_test", X_test[0])
print("Sample of y_test", y_test[0])


# In[70]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD


# In[71]:


np.random.seed(666)


# In[97]:


model = Sequential()

model.add(Dense(1, input_dim=13, activation='relu'))
model.add(Dense(1, activation='linear'))

opt = SGD(lr=0.01, momentum=0.9)
model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mse'])

model.summary()


# In[98]:


history = model.fit(X_train, y_train, epochs=100, verbose=False);


# In[99]:


score = model.evaluate(X_test, y_test)
print(score)


# In[54]:


import matplotlib.pyplot as plt


# In[31]:


plt.plot(history.history['loss'], label='train');


# ### Linear Regression

# In[55]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# In[56]:


line = LinearRegression()
line.fit(X_train, y_train)
predict = line.predict(X_test)

mse = mean_squared_error(y_test, predict)
print(mse)


# ### Back to NN

# In[107]:


# feature engineering
from sklearn.decomposition import PCA

pca = PCA(n_components=6)
principalComponents = pca.fit_transform(X_train)
testComponents = pca.transform(X_test)
train_feat = pd.DataFrame(data = principalComponents, columns = ['pc_1', 'pc_2', 'pc_3', 'pc_4', 'pc_5', 'pc_6'])
test_feat = pd.DataFrame(data = testComponents, columns = ['pc_1', 'pc_2', 'pc_3', 'pc_4', 'pc_5', 'pc_6'])


# In[110]:


train_feat.head()


# In[108]:


model2 = Sequential()

model2.add(Dense(3, input_dim=6, activation='relu'))
model2.add(Dense(1, activation='linear'))

opt = SGD(lr=0.01, momentum=0.9)
model2.compile(loss='mean_squared_error', optimizer=opt, metrics=['mse'])

model2.summary()


# In[112]:


history = model2.fit(train_feat.values, y_train, epochs=100, verbose=False);


# In[114]:


score = model2.evaluate(test_feat.values, y_test)
print(score)


# ### Back to Linear

# In[116]:


line2 = LinearRegression()
line2.fit(train_feat.values, y_train)
predict = line2.predict(test_feat.values)

mse = mean_squared_error(y_test, predict)
print(mse)


# ### Much worse for both

# ## Use the Keras Library to build an image recognition network using the Fashion-MNIST dataset (also comes with keras)
# 
# - Load and preprocess the image data similar to how we preprocessed the MNIST data in class.
# - Make sure to one-hot encode your category labels
# - Make sure to have your final layer have as many nodes as the number of classes that you want to predict.
# - Try different hyperparameters. What is the highest accuracy that you are able to achieve.
# - Use the history object that is returned from model.fit to make graphs of the model's loss or train/validation accuracies by epoch. 
# - Remember that neural networks fall prey to randomness so you may need to run your model multiple times (or use Cross Validation) in order to tell if a change to a hyperparameter is truly producing better results.

# In[120]:


from tensorflow.keras.datasets import fashion_mnist


# In[154]:


(train_x, train_y), (test_x, test_y) = fashion_mnist.load_data()


# In[123]:


train_x.shape, train_y.shape, test_x.shape, test_y.shape


# In[128]:


plt.imshow(train_x[1000]); # Take a look at some of the data


# In[133]:


from tensorflow.keras.utils import to_categorical


# In[155]:


num_classes = 10
train_y = to_categorical(train_y, num_classes)
test_y = to_categorical(test_y, num_classes)


# In[157]:


train_x = train_x.reshape(60000, -1).astype('float32')
test_x = test_x.reshape(10000, -1).astype('float32')


# In[165]:


model3 = Sequential()

model3.add(Dense(1024, input_dim=784, activation='relu'))
model3.add(Dense(10, activation='softmax'))

opt = SGD(lr=0.01, momentum=0.9)
model3.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model3.summary()


# In[167]:


history = model3.fit(train_x, train_y, epochs=15, batch_size=100);


# In[170]:


score = model3.evaluate(test_x, test_y, verbose=0)
print(score[1])


# ## Stretch Goals:
# 
# - Use Hyperparameter Tuning to make the accuracy of your models as high as possible. (error as low as possible)
# - Use Cross Validation techniques to get more consistent results with your model.
# - Use GridSearchCV to try different combinations of hyperparameters. 
# - Start looking into other types of Keras layers for CNNs and RNNs maybe try and build a CNN model for fashion-MNIST to see how the results compare.
