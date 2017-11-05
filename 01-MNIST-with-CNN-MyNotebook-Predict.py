
# coding: utf-8

# # MNIST with CNN

# In[1]:


import os
import urllib

import numpy as np
import tensorflow as tf

from skimage import io
from skimage import transform as tform
from skimage.color import rgb2gray, rgb2grey, rgba2rgb


# ## Init variables

# In[2]:


savefile = "./STORED_model/my_trained_model.json"


# ### Helper Functions

# In[3]:


def init_weights(shape):
    init_random_dist = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init_random_dist)


# In[4]:


def init_bias(shape):
    init_bias_vals = tf.constant(0.1, shape=shape)
    return tf.Variable(init_bias_vals)


# In[5]:


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# In[6]:


def max_pool_2by2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


# In[7]:


def convolutional_layer(input_x, shape):
    W = init_weights(shape)
    b = init_bias([shape[3]])
    return tf.nn.relu(conv2d(input_x, W) + b)


# In[8]:


def normal_full_layer(input_layer, size):
    input_size = int(input_layer.get_shape()[1])
    W = init_weights([input_size, size])
    b = init_bias([size])
    return tf.matmul(input_layer, W) + b


# # Helper (custom functions)

# In[9]:


def one_hot_encode(pos):
    out = np.zeros(10)
    out[pos] = 1
    return out


# ### Placeholders

# In[10]:


x = tf.placeholder(tf.float32,shape=[None,784])


# In[11]:


y_true = tf.placeholder(tf.float32,shape=[None,10])


# ### Layers

# In[12]:


x_image = tf.reshape(x,[-1,28,28,1])


# In[13]:


# Using a 6by6 filter here, used 5by5 in video, you can play around with the filter size
# You can change the 32 output, that essentially represents the amount of filters used
# You need to pass in 32 to the next input though, the 1 comes from the original input of 
# a single image.
convo_1 = convolutional_layer(x_image,shape=[6,6,1,32])
convo_1_pooling = max_pool_2by2(convo_1)


# In[14]:


# Using a 6by6 filter here, used 5by5 in video, you can play around with the filter size
# You can actually change the 64 output if you want, you can think of that as a representation
# of the amount of 6by6 filters used.
convo_2 = convolutional_layer(convo_1_pooling,shape=[6,6,32,64])
convo_2_pooling = max_pool_2by2(convo_2)


# In[15]:


# Why 7 by 7 image? Because we did 2 pooling layers, so (28/2)/2 = 7
# 64 then just comes from the output of the previous Convolution
convo_2_flat = tf.reshape(convo_2_pooling,[-1,7*7*64])
full_layer_one = tf.nn.relu(normal_full_layer(convo_2_flat,1024))


# In[16]:


# NOTE THE PLACEHOLDER HERE!
hold_prob = tf.placeholder(tf.float32)
full_one_dropout = tf.nn.dropout(full_layer_one,keep_prob=hold_prob)


# In[17]:


y_pred = normal_full_layer(full_one_dropout,10)


# ### Loss Function

# In[18]:


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y_pred))


# ### Optimizer

# In[19]:


optimizer = tf.train.AdamOptimizer(learning_rate=0.0001) # 0.0001
train = optimizer.minimize(cross_entropy)


# ### Intialize Variables

# In[20]:


init = tf.global_variables_initializer()
saver = tf.train.Saver()


# ### Session

# In[21]:


imageURL = 'https://firebasestorage.googleapis.com/v0/b/tensorweb-af554.appspot.com/o/test01.png?alt=media&token=26c1db9f-b36b-420d-acce-48845753b9f9'
imageURL2 = 'https://firebasestorage.googleapis.com/v0/b/tensorweb-af554.appspot.com/o/testcar01.png?alt=media&token=a29a9ccd-1cd1-4731-9f17-7bab3a4bda98'
imageURL3 = 'https://firebasestorage.googleapis.com/v0/b/tensorweb-af554.appspot.com/o/test03.png?alt=media&token=d235eba9-d645-48f4-8616-825e19d35de6'
imageURL4 = 'https://firebasestorage.googleapis.com/v0/b/tensorweb-af554.appspot.com/o/test04.png?alt=media&token=b3a33d7e-9189-4255-9023-6ce6ba2d77df'


# In[22]:


def predictFromUrlImage(imageUrl):
    with tf.Session() as sess:
        # restore the model
        saver.restore(sess, "./STORED_model/my_trained_model.json")
        
        imageData = readGrayImageFromUrl(imageUrl)
        imageSimple = simplifyImage(imageData)
        
        feed_dict = {x: imageSimple, y_true: np.zeros((1, 10)), hold_prob : 0.5 }
        classification = sess.run(tf.argmax(y_pred,1), feed_dict)
        
    return classification


# In[23]:


def readGrayImageFromUrl(url):
    imageToPredict = rgba2rgb(io.imread(url))
    return rgb2grey(imageToPredict)


# In[24]:


def simplifyImage(originalImage):
    partialResizeImage = tform.resize(originalImage, (28, 28), order=5)
    return (1 - np.reshape(partialResizeImage,newshape=(1,784)))


# In[29]:


myPrediction = predictFromUrlImage(imageURL4)
print(myPrediction)
# print(sih.fileList[int(myPrediction)])


# ## Great Job!
