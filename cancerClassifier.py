"""
    @author: Andrew Kulpa

    The following is a tensorflow program using a breast cancer data set from UCI.
    The program implements a fully connected neural network with an Adam optimizer
    and an exponentially decaying learning rate.  
"""
import tensorflow as tf
import numpy as np
import pandas as pd
tf.logging.set_verbosity(tf.logging.INFO)

# Generate training input from the csv file.
trainX = pd.read_csv("wdbc_train.data", header=None, usecols=[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31])
trainX['buffer1'] = 0
trainX['buffer2'] = 0
trainX['buffer3'] = 0
trainX['buffer4'] = 0
trainX['buffer5'] = 0
trainX['buffer6'] = 0
trainX = np.array(trainX).astype(np.float32) # trainX is 499x36

labels = pd.read_csv("wdbc_train.data", header=None, usecols=[1]) # Read the cancer classifications
encoded_labels = np.array(pd.get_dummies(labels)) # one hot encode; train labels are of shape 499x2 ([0, 1] = 'M', [1,0] = 'B')

# Generate test input from the csv file.
testX = pd.read_csv("wdbc_test.data", header=None, usecols=[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31])
testX['buffer1'] = 0
testX['buffer2'] = 0
testX['buffer3'] = 0
testX['buffer4'] = 0
testX['buffer5'] = 0
testX['buffer6'] = 0
testX = np.array(testX).astype(np.float32) # testX is 70x36

test_labels = pd.read_csv("wdbc_test.data", header=None, usecols=[1])
test_encoded_labels = np.array(pd.get_dummies(test_labels)) # one hot encode; train labels are of shape 70x2 ([0, 1] = 'M', [1,0] = 'B')

# Parameters
initial_learning_rate = 0.1
training_epochs = 100000
decay_steps = 100
display_epoch = 10000
decay_base_rate = 0.96


# Network
hidden_1_nodes = 100
hidden_2_nodes = 100
input_nodes = 36
classes = 2

# Create placeholder tensors
X = tf.placeholder("float", [None, input_nodes])
Y = tf.placeholder("float", [None, classes])

# Create hashes of weights and biases, shaped to conform to the inputs and preceding layers
weights = {
    'hidden1': tf.Variable(tf.random_normal([input_nodes, hidden_1_nodes])),
    'hidden2': tf.Variable(tf.random_normal([hidden_1_nodes, hidden_2_nodes])),
    'output': tf.Variable(tf.random_normal([hidden_2_nodes, classes])),
}
biases = {
    'bias1': tf.Variable(tf.random_normal([hidden_1_nodes])),
    'bias2': tf.Variable(tf.random_normal([hidden_2_nodes])),
    'output': tf.Variable(tf.random_normal([classes])),
}

# Create the model for the multi-layered neural network, utilizing the previous weights and biases.
def mlnn(inX):
    hidden1 = tf.add(tf.matmul(inX, weights['hidden1']), biases['bias1'])
    hidden2 = tf.add(tf.matmul(hidden1, weights['hidden2']), biases['bias2'])
    logits = tf.add(tf.matmul(hidden2, weights['output']), biases['output'])
    return logits

# Define the model, loss, optimizer, and learning rate
logits = mlnn(X)
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
global_step = tf.Variable(0, trainable=False) # https://www.tensorflow.org/api_docs/python/tf/train/exponential_decay
learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step, decay_steps, decay_base_rate, staircase=True)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

train_op = optimizer.minimize(loss_op, global_step = global_step)

# Initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    # Training
    for epoch in range(training_epochs):
        # Backpropagation optimization and cost operation
        _, cost = sess.run([train_op, loss_op], feed_dict={X: trainX,
                                                        Y: encoded_labels})
        # Display results for each epoch cycle
        if epoch % display_epoch == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(cost))
    print("Training done!")
    # Apply softmax to logits
    pred = tf.nn.softmax(logits)  
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    # Calculate the accuracy of the model
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Test Data Accuracy:", accuracy.eval({X: testX, Y: test_encoded_labels}))
