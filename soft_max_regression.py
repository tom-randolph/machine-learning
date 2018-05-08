'''In this script, we create a simple model to classify the MNIST data-set.
This is based on the tutorial here:
https://www.tensorflow.org/versions/r1.1/get_started/mnist/beginners
This is a "softmax regression", which is essential a NN of a single layer,
using softmax as the activation function and gradient descent to optimize
the cross-entropy cost function'''

print("Importing modules")
# import tensorflow and the MNIST data
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# the dimenstion of the MNIST images is 28px28p
dim_img=28

# and the classes of the ouput range from 0-9, making 10 total different classess
num_classes=10

print("Importing training data")
# here we import the image data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

print("Initializing Variables")
# create a place holder for the the (flattened) pixel data
x = tf.placeholder(tf.float32, [None,dim_img**2])

# and variables for the weights and biases
W = tf.Variable(tf.random_normal([dim_img**2,num_classes]))
b = tf.Variable(tf.random_normal([num_classes]))

# y is the "unactivated" output of the 1st and only layer
y=tf.matmul(x,W)+b

# y_ is the actual value of class (0-9) in one-hot form
y_ = tf.placeholder(tf.float32,[None,num_classes])

# we use a cross-etropy after softmax cost function
cross_entropy= tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y))

# and use Gradient Descent to minimize this function with a learning rate of .5
train_step = tf.train.GradientDescentOptimizer(.75).minimize(cross_entropy)

# alternitvely, try using the Adam optimization method
# train_step = tf.train.AdamOptimizer(.7).minimize(cross_entropy)

# initialize the variables
init=tf.global_variables_initializer()

# and start the session
sess=tf.Session()
sess.run(init)

# will train 1000 times on seperate batches of 550 images, which for the MNIST data set should constitute 1 epoch
print("Training the model")
for _ in range(1000):
    xs, ys = mnist.train.next_batch(550)

    sess.run(train_step, feed_dict={x:xs,y_:ys})

correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print("Determining accuracy on test images...")
print(sess.run(accuracy,feed_dict={x:mnist.test.images,
                                    y_:mnist.test.labels}))
print("Done!")
