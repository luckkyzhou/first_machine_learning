from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range

batch_size = 128
image_size = 28
num_labels = 10

pickle_file = 'notMNIST.pickle'

with open(pickle_file,'rb') as f:
    save = pickle.load(f)
    train_dataset = save['train_dataset']
    train_labels = save['train_labels']
    valid_dataset = save['valid_dataset']
    valid_labels = save['valid_labels']
    test_dataset = save['test_dataset']
    test_labels = save['test_labels']
    del save

def accuracy(predictions,labels):
    return (100.0*np.sum(np.argmax(predictions,1) == np.argmax(labels,1))/predictions.shape[0])

def reformat(dataset,labels):
    dataset = dataset.reshape((-1,image_size*image_size)).astype(np.float32)
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return dataset,labels
train_dataset,train_labels = reformat(train_dataset,train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)

def inference(images, hidden1_units, hidden2_units):
    with tf.name_scope('hidden1'):
        weights = tf.Variable(tf.truncated_normal([image_size*image_size, hidden1_units]))
        biases = tf.Variable(tf.zeros([hidden1_units]))
        hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)
    with tf.name_scope('hidden2'):
        weights = tf.Variable(tf.truncated_normal([hidden1_units, hidden2_units]))
        biases = tf.Variable(tf.zeros([hidden2_units]))
        hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
    with tf.name_scope('softmax_linear'):
        weights = tf.Variable(tf.truncated_normal([hidden2_units, num_labels]))
        biases = tf.Variable(tf.zeros(num_labels))
        logits = tf.matmul(hidden2, weights) + biases
    return logits

graph = tf.Graph()
with graph.as_default():
    tf_train_dataset = tf.placeholder(tf.float32,shape=(batch_size,image_size*image_size))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)


    weights = tf.Variable(tf.truncated_normal([image_size*image_size,num_labels]))
    biases = tf.Variable(tf.zeros([num_labels]))

    logits = inference(tf_train_dataset, 5, 2)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels,logits=logits))
    #regulation_loss = loss + 0.35*tf.nn.l2_loss(weights) / batch_size

    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(0.5, global_step, 100, 0.96, staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(tf.matmul(tf_valid_dataset, weights) + biases)
    test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)

num_steps = 3001
with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print("Initialized")
    for step in range(num_steps):
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 500 == 0):
            print("Minibatch loss at step %d: %f" % (step, l))
            print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
            print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), valid_labels))
    print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))