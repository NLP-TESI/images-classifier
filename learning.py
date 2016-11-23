
# coding: utf-8

# In[1]:

from load_hist import *

import numpy as np
import tensorflow as tf


# In[2]:

# help function to sampling data
def get_sample(num_samples, X_data, y_data):
    positions = np.arange(len(y_data))
    np.random.shuffle(positions)

    X_sample = []
    y_sample = []

    for posi in positions[:num_samples]:
        X_sample.append(X_data[posi])
        y_sample.append(y_data[posi])

    return X_sample, y_sample


# In[3]:

######################## creating the model architecture #######################################

input_size = len(X_train[0])
label_size = len(y_train[0])

print "input size: ", input_size, ", label size: ", label_size

# input placeholder
x = tf.placeholder(tf.float32, [None, input_size])

# output placeholder
y_ = tf.placeholder(tf.float32, [None, label_size])

num_nodes_layer1 = 500
num_nodes_layer2 = 300

# weights of the neurons
W1 = tf.Variable(tf.random_normal([input_size, num_nodes_layer1], stddev=35))
b1 = tf.Variable(tf.random_normal([num_nodes_layer1], stddev=35))


# weights of the neurons in second layer
W2 = tf.Variable(tf.random_normal([num_nodes_layer1,num_nodes_layer2], stddev=0.35))
b2 = tf.Variable(tf.random_normal([num_nodes_layer2], stddev=0.35))


# weights of the neurons in third layer
W3 = tf.Variable(tf.random_normal([num_nodes_layer2,label_size], stddev=0.35))
b3 = tf.Variable(tf.random_normal([label_size], stddev=0.35))

# output of the network
layer1 = tf.nn.softmax(tf.matmul(x, W1) + b1)
layer2 = tf.nn.softmax(tf.matmul(layer1, W2) + b2)
y_estimated = tf.nn.softmax(tf.matmul(layer2, W3) + b3)


# function to measure the error
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_estimated), reduction_indices=[1]))


# how to train the model
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


# how to evaluate the model
correct_prediction = tf.equal(tf.argmax(y_estimated,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# In[4]:

######################## training the model #######################################

# applying a value for each variable (in this case W and b)
init = tf.initialize_all_variables()


# a session is dependent of the enviroment where tensorflow is running
sess = tf.Session()
sess.run(init)



num_batch_trainning = 100
for i in range(10000): # trainning 1000 times

    # randomizing positions
    X_sample, y_sample = get_sample(num_batch_trainning, X_train, y_train)

    # where the magic happening
    sess.run(train_step, feed_dict={x: X_sample, y_:  y_sample})

    # print the accuracy result
    if i % 10 == 0:
        print "\r",i, ": ", (sess.run(accuracy, feed_dict={x: X_validation, y_: y_validation})),


print "\n\n\n"
print "TEST RESULT: ", (sess.run(accuracy, feed_dict={x: X_test, y_: y_test}))


# In[ ]:



