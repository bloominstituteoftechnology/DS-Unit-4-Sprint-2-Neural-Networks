#!/usr/bin/env python
# coding: utf-8

# Lambda School Data Science
# 
# *Unit 4, Sprint 2, Module 3*
# 
# ---

# # Neural Network Frameworks (Prepare)

# ## Learning Objectives
# * <a href="#p1">Part 1</a>: Introduce the Keras Sequential Model API
# * <a href="#p2">Part 2</a>: Learn How to Select Model Architecture 
# * <a href="#p3">Part 3</a>: Discuss the trade-off between various activation functions
# 
# ## Lets Use Libraries!
# 
# The objective of the last two days has been to familiarize you with the fundamentals of neural networks: terminology, structure of networks, forward propagation, error/cost functions, backpropagation, epochs, and gradient descent. We have tried to reinforce these topics by requiring to you code some of the simplest neural networks by hand including Perceptrons (single node neural networks) and Multi-Layer Perceptrons also known as Feed-Forward Neural Networks. Continuing to do things by hand would not be the best use of our limited time. You're ready to graduate from doing things by hand and start using some powerful libraries to build cutting-edge predictive models. 

# # Keras Sequential API (Learn)

# ## Overview
# 
# > "Keras is a high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano. It was developed with a focus on enabling fast experimentation. Being able to go from idea to result with the least possible delay is key to doing good research. Use Keras if you need a deep learning library that:
# 
# > Allows for easy and fast prototyping (through user friendliness, modularity, and extensibility).
# Supports both convolutional networks and recurrent networks, as well as combinations of the two.
# Runs seamlessly on CPU and GPU." 

# ### Keras Perceptron Sample

# In[1]:


import pandas as pd

data = { 'x1': [0,1,0,1],
         'x2': [0,0,1,1],
         'y':  [1,1,1,0]
       }

df = pd.DataFrame.from_dict(data).astype('int')
X = df[['x1', 'x2']].values
y = df['y'].values


# In[8]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# This is our perceptron from Monday's by-hand: 
model = Sequential()
model.add(Dense(1,input_dim=2, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X,y, epochs=5);


# In[9]:


# evaluate the model
scores = model.evaluate(X, y)
print(f"{model.metrics_names[1]}: {scores[1]*100}")


# ## Follow Along
# 
# In the `Sequential` api model, you specify a model architecture by 'sequentially specifying layers. This type of specification works well for feed forward neural networks in which the data flows in one direction (forward propagation) and the error flows in the opposite direction (backwards propagation). The Keras `Sequential` API follows a standardarized worklow to estimate a 'net: 
# 
# 1. Load Data
# 2. Define Model
# 3. Compile Model
# 4. Fit Model
# 5. Evaluate Model
# 
# You saw these steps in our Keras Perceptron Sample, but let's walk thru each step in detail.

# ### Load Data
# 
# Our life is going to be easier if our data is already cleaned up and numeric, so lets use this dataset from Jason Brownlee that is already numeric and has no column headers so we'll need to slice off the last column of data to act as our y values.

# In[10]:


import pandas as pd

url ="https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

dataset = pd.read_csv(url, header=None)


# In[11]:


dataset.head()


# In[12]:


X = dataset.values[:,0:8]
print(X.shape)
print(X)


# In[13]:


y = dataset.values[:,-1]
print(y.shape)
print(y)


# ### Define Model

# In[14]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy as np

np.random.seed(812)


# I'll instantiate my model as a "sequential" model. This just means that I'm going to tell Keras what my model's architecture should be one layer at a time.

# In[15]:


# https://keras.io/getting-started/sequential-model-guide/
model = Sequential()


# Adding a "Dense" layer to our model is how we add "vanilla" perceptron-based layers to our neural network. These are also called "fully-connected" or "densely-connected" layers. They're used as a layer type in lots of other Neural Net Architectures but they're not referred to as perceptrons or multi-layer perceptrons very often in those situations even though that's what they are.
# 
#  > ["Just your regular densely-connected NN layer."](https://keras.io/layers/core/)
#  
#  The first argument is how many neurons we want to have in that layer. To create a perceptron model we will just set it to 1. We will tell it that there will be 8 inputs coming into this layer from our dataset and set it to use the sigmoid activation function.

# In[16]:


model.add(Dense(1, input_dim=8, activation="sigmoid")) #Relu is valid option. 


# ### Compile Model
# Using binary_crossentropy as the loss function here is just telling keras that I'm doing binary classification so that it can use the appropriate loss function accordingly. If we were predicting non-binary categories we might assign something like `categorical_crossentropy`. We're also telling keras that we want it to report model accuracy as our main error metric for each epoch. We will also be able to see the overall accuracy once the model has finished training.
# 
# #### Adam Optimizer
# Check out this links for more background on the Adam optimizer and Stohastic Gradient Descent
# * [Adam Optimization Algorithm](https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/)
# * [Adam Optimizer - original paper](https://arxiv.org/abs/1412.6980)

# In[33]:


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# ### Fit Model
# 
# Lets train it up! `model.fit()` has a `batch_size` parameter that we can use if we want to do mini-batch epochs, but since this tabular dataset is pretty small we're just going to delete that parameter. Keras' default `batch_size` is `None` so omiting it will tell Keras to do batch epochs.

# In[35]:


model.fit(X, y, epochs=150);


# In[20]:


y.shape


# ### Evaluate Model

# In[21]:


y[:50]


# In[36]:


# Predicting never diabetes
sum(y) / len(y) 


# In[37]:


scores = model.evaluate(X,y)
print(f"{model.metrics_names[1]}: {scores[1]*100}")


# ### Unstable Results
# 
# You'll notice that if we rerun the results might differ from the origin run. This can be explain by a bunch of factors. Check out some of them in this article: 
# 
# <https://machinelearningmastery.com/randomness-in-machine-learning/>

# ## Challenge
# 
# You will be expected to leverage the Keras `Sequential`api to estimate a feed forward neural networks on a dataset.

# # Choosing Architecture (Learn)

# ## Overview
# 
# Choosing an architecture for a neural network is almost more an art than a science. Let's do a few experiments:

# ## Follow Along

# In[47]:


# Tell me your ideas

model_improved = Sequential()

model_improved.add(Dense(4, input_dim=8, activation='relu'))
model_improved.add(Dense(3, activation='relu'))
model_improved.add(Dense(1, activation='sigmoid'))

model_improved.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy'])

# Let's inspect our new architecture
model_improved.summary()


# In[48]:


model_improved.fit(X, y, epochs=150, verbose=False); # What parameters can I specify here? 
# verbose stops the text from printing


# In[49]:


score = model_improved.evaluate(X,y)
# print(f"{model_improved.metrics_names[1]}: {scores[1]*100}")
print(score[1])


# ### Experiment 1

# In[60]:


# Tell me your ideas

model_improved = Sequential()

model_improved.add(Dense(4, input_dim=8, activation='relu'))
model_improved.add(Dense(3, activation='relu'))
model_improved.add(Dense(3, activation='relu'))
model_improved.add(Dense(1, activation='sigmoid'))

model_improved.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy'])

# Let's inspect our new architecture
model_improved.summary()


# In[61]:


model_improved.fit(X, y, epochs=150, verbose=False); # What parameters can I specify here? 
# verbose stops the text from printing


# In[62]:


score = model_improved.evaluate(X,y)
# print(f"{model_improved.metrics_names[1]}: {scores[1]*100}")
print(score[1])


# ### Experiment 2

# In[56]:


# Tell me your ideas

model_improved = Sequential()

model_improved.add(Dense(8, input_dim=8, activation='relu', name="Dense1"))
model_improved.add(Dense(8, activation='relu'))
model_improved.add(Dense(8, activation='relu'))
model_improved.add(Dense(1, activation='sigmoid'))

model_improved.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy'])

# Let's inspect our new architecture
model_improved.summary()


# In[54]:


model_improved.fit(X, y, epochs=150, verbose=False); # What parameters can I specify here? 
# verbose stops the text from printing


# In[55]:


score = model_improved.evaluate(X,y)
# print(f"{model_improved.metrics_names[1]}: {scores[1]*100}")
print(score[1])


# ## Challenge
# 
# You will have to choose your own architectures in today's module project. 

# # Activation Functions (Learn)

# ## Overview
# What is an activation function and how does it work?
# 
# - Takes in a weighted sum of inputs + a bias from the previous layer and outputs an "activation" value.
# - Based its inputs the neuron decides how 'activated' it should be. This can be thought of as the neuron deciding how strongly to fire. You can also think of it as if the neuron is deciding how much of the signal that it has received to pass onto the next layer. 
# - Our choice of activation function does not only affect signal that is passed forward but also affects the backpropagation algorithm. It affects how we update weights in reverse order since activated weight/input sums become the inputs of the next layer. 

# ## Follow Along

# ### Step Function
# 
# ![Heaviside Step Function](https://upload.wikimedia.org/wikipedia/commons/thumb/d/d9/Dirac_distribution_CDF.svg/325px-Dirac_distribution_CDF.svg.png)
# 
# All or nothing, a little extreme, which is fine, but makes updating weights through backpropagation impossible. Why? remember that during backpropagation we use derivatives in order to determine how much to update or not update weights. What is the derivative of the step function?

# ### Linear Function
# 
# ![Linear Function](http://www.roconnell.net/Parent%20function/linear.gif)
# 
# The linear function takes the opposite tact from the step function and passes the signal onto the next layer by a constant factor. There are problems with this but the biggest problems again lie in backpropagation. The derivative of any linear function is a horizontal line which would indicate that we should update all weights by a constant amount every time -which on balance wouldn't change the behavior of our network. Linear functions are typically only used for very simple tasks where interpretability is important, but if interpretability is your highest priority, you probably shouldn't be using neural networks in the first place.

# ### Sigmoid Function
# 
# ![Sigmoid Function](https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/Logistic-curve.svg/480px-Logistic-curve.svg.png)
# 
# The sigmoid function works great as an activation function! it's continuously differentiable, its derivative doesn't have a constant slope, and having the higher slope in the middle pushes y value predictions towards extremes which is particularly useful for binary classification problems. I mean, this is why we use it as the squishifier in logistic regression as well. It constrains output, but over repeated epochs pushes predictions towards a strong binary prediction. 
# 
# What's the biggest problem with the sigmoid function? The fact that its slope gets pretty flat so quickly after its departure from zero. This means that updating weights based on its gradient really diminishes the size of our weight updates as our model gets more confident about its classifications. This is why even after so many iterations with our test score example we couldn't reach the levels of fit that our gradient descent based model could reach in just a few epochs.

# ### Tanh Function
# 
# ![Tanh Function](http://mathworld.wolfram.com/images/interactive/TanhReal.gif)
# 
# What if the sigmoid function didn't get so flat quite as soon when moving away from zero and was a little bit steeper in the middle? That's basically the Tanh function. The Tanh function can actually be created by scaling the sigmoid function by 2 in the y dimension and subtracting 1 from all values. It has basically the same properties as the sigmoid, still struggles from diminishingly flat gradients as we move away from 0, but its derivative is higher around 0 causing weights to move to the extremes a little faster. 

# ### ReLU Function
# 
# ![ReLU Function](https://cdn-images-1.medium.com/max/937/1*oePAhrm74RNnNEolprmTaQ.png)
# 
# ReLU stands for Rectified Linear Units it is by far the most commonly used activation function in modern neural networks. It doesn't activate neurons that are being passed a negative signal and passes on positive signals. Think about why this might be useful. Remember how a lot of our initial weights got set to negative numbers by chance? This would have dealt with those negative weights a lot faster than the sigmoid function updating. What does the derivative of this function look like? It looks like the step function! This means that not all neurons are activated. With sigmoid basically all of our neurons are passing some amount of signal even if it's small making it hard for the network to differentiate important and less important connections. ReLU turns off a portion of our less important neurons which decreases computational load, but also helps the network learn what the most important connections are faster. 
# 
# What's the problem with relu? Well the left half of its derivative function shows that for neurons that are initialized with weights that cause them to have no activation, our gradient will not update those neuron's weights, this can lead to dead neurons that never fire and whose weights never get updated. We would probably want to update the weights of neurons that didn't fire even if it's just by a little bit in case we got unlucky with our initial weights and want to give those neurons a chance of turning back on in the future.

# ### Leaky ReLU
# 
# ![Leaky ReLU](https://cdn-images-1.medium.com/max/1600/1*ypsvQH7kvtI2BhzR2eT_Sw.png)
# 
# Leaky ReLU accomplishes exactly that! it avoids having a gradient of 0 on the left side of its derivative function. This means that even "dead" neurons have a chance of being revived over enough iterations. In some specifications the slope of the leaky left-hand side can also be experimented with as a hyperparameter of the model!

# ### Softmax Function
# 
# ![Softmax Function](https://cdn-images-1.medium.com/max/800/1*670CdxchunD-yAuUWdI7Bw.png)
# 
# Like the sigmoid function but more useful for multi-class classification problems. The softmax function can take any set of inputs and translate them into probabilities that sum up to 1. This means that we can throw any list of outputs at it and it will translate them into probabilities, this is extremely useful for multi-class classification problems. Like MNIST for example...

# ### Major takeaways
# 
# - ReLU is generally better at obtaining the optimal model fit.
# - Sigmoid and its derivatives are usually better at classification problems.
# - Softmax for multi-class classification problems. 
# 
# You'll typically see ReLU used for all initial layers and then the final layer being sigmoid or softmax for classification problems. But you can experiment and tune these selections as hyperparameters as well!

# ### MNIST with Keras 
# 
# #### This will be a good chance to bring up dropout regularization. :)

# In[63]:


### Let's do it!

from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Stretch - use dropout 
import numpy as np


# In[64]:


# Hyper Parameters
batch_size = 64
num_classes = 10
epochs = 20


# In[65]:


# Load the Data
(X_train, y_train), (X_test, y_test) = mnist.load_data()


# In[66]:


X_train[0].shape


# In[67]:


X_train.shape


# In[68]:


X_train[0]


# In[69]:


# Reshape the data
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)


# In[70]:


# X Variable Types
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')


# In[71]:


y_train[2] 


# In[74]:


# Correct Encoding on Y
# What softmax expects = [0,0,0,0,0,1,0,0,0,0]

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# In[75]:


y_train[2]


# In[76]:


mnist_model = Sequential()

# Input => Hidden
mnist_model.add(Dense(16, input_dim=784, activation='relu'))
# Hidden
mnist_model.add(Dense(16, activation='relu'))
# Hidden
mnist_model.add(Dense(16, activation='relu'))
# Hidden
mnist_model.add(Dense(16, activation='relu'))
# Output
mnist_model.add(Dense(10,activation='softmax'))

#Compile
mnist_model.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])

mnist_model.summary()


# In[77]:


16 *  784


# In[78]:


y_test.shape


# In[79]:


history = mnist_model.fit(X_train, y_train, batch_size=32, epochs=100, verbose=False)
scores = mnist_model.evaluate(X_test, y_test)
#print(f'{mnist_model.metrics_names[1]}: {scores[1]*100}')


# ### Dropout Regularization
# 
# ![Regularization](https://upload.wikimedia.org/wikipedia/commons/thumb/0/02/Regularization.svg/354px-Regularization.svg.png)

# In[80]:


### Let's do it!
from tensorflow import keras 
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

import numpy as np

mnist_model = Sequential()

# Hidden
mnist_model.add(Dense(32, input_dim=784, activation='relu'))
mnist_model.add(Dropout(0.2))
mnist_model.add(Dense(16, activation='relu'))
mnist_model.add(Dropout(0.2))
# Output Layer
mnist_model.add(Dense(10, activation='softmax'))

mnist_model.compile(loss='categorical_crossentropy',
                    optimizer='adam', 
                    metrics=['accuracy'])
mnist_model.summary()


# In[81]:


history = mnist_model.fit(X_train, y_train, batch_size=32, epochs=epochs, validation_split=.1, verbose=0)
scores = mnist_model.evaluate(X_test, y_test)
print(f'{mnist_model.metrics_names[1]}: {scores[1]*100}')


# ## Challenge
# 
# You will apply your choice of activation function inside two Keras Seqeuntial models today. 
