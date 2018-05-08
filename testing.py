import tensorflow as tf
import numpy as np

'''A simple linear regession model in tensorflow inspired by this tutrial:
https://medium.com/all-of-us-are-belong-to-machines/the-gentlest-introduction-to-tensorflow-248dc871a224'''


# setup placeholders for x and y values
x=tf.placeholder(tf.float32,[None,1])
y_=tf.placeholder(tf.float32,[None,1])

# setup Weight and Bias variables
W=tf.Variable(tf.zeros([1,1]))

b=tf.Variable(tf.zeros([1]))

# libnear forward progagation function
y=tf.matmul(x,W)+b

# least mean squares cost function
cost=tf.reduce_mean(tf.square((y_-y)))

# optimization method
train_step=tf.train.GradientDescentOptimizer(.000001).minimize(cost)

# initialize the variables
init=tf.global_variables_initializer()

# and start the session
sess=tf.Session()
sess.run(init)
runs=4000
Ws=np.zeros(epochs)
bs=np.zeros(epochs)
for i in range(runs):

    # values are simply model with a slope of ~6 with some normal random variance
    xs=np.array([[i]])
    ys=np.array([[np.random.normal()+6*i]])

    # train the model
    sess.run(train_step,feed_dict={x:xs,y_:ys})

    print("After %d iterations: " %i )
    print("W: %f" %sess.run(W))
    print("b: %f" %sess.run(b))

    # and calculate the cost
    cost_=sess.run(cost,feed_dict={x:xs,y_:ys})

    # if the cost is very small (~.0001)
    print("cost: %f" %float(cost_))
    if cost_<.0001 and i>10:break
