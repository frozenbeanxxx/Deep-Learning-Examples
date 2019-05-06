import tensorflow as tf 
import numpy as np

def CrossEntropyTest():
    logits = tf.constant([[1.0,2.0,3.0],\
                        [1.0,2.0,3.0],\
                        [1.0,2.0,3.0]])

    y = tf.nn.softmax(logits)

    y_=tf.constant([[0.0,0.0,1.0],\
                    [0.0,0.0,1.0],\
                    [0.0,0.0,1.0]])

    cross_entropy = -tf.reduce_sum(y_*tf.log(y))

    cross_entropy2 = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_))

    with tf.Session() as sess:
        softmax=sess.run(y)
        c_e = sess.run(cross_entropy)
        c_e2 = sess.run(cross_entropy2)

        print(softmax)
        print(c_e)
        print(c_e2)

def TfConvTest():
    #Shape must be rank 4 but is rank 2 for 'Conv2D' (op: 'Conv2D') with input shapes: [3,3], [3,3].
    #conv2d input must be rank4
    X = tf.constant([[[[1.0,2.0,3.0],\
                    [1.0,2.0,3.0],\
                    [1.0,2.0,3.0]]]])
    W = tf.constant([[[[1.0,2.0,3.0],\
                    [1.0,2.0,3.0],\
                    [1.0,2.0,3.0]]]])
    W1 = tf.constant([[[[1.0]]]])
    strides = [1, 1, 1, 1]
    padding = "SAME"
    L = tf.nn.conv2d(X, W, strides=strides, padding=padding)
    with tf.Session() as sess:
        b = sess.run(L)
        print(b)
        c = sess.run(tf.shape(X))
        print(c)

    X2 = tf.ones([1, 4, 4, 1])
    W2 = tf.ones([1, 1, 1, 2])
    strides2 = [1, 1, 1, 1]
    L2 = tf.nn.conv2d(X2, W2, strides=strides2, padding=padding)
    with tf.Session() as sess:
        #print(sess.run(X2))
        print(sess.run(L2))

def GradientTest():
    sess = tf.Session()

    x_input = tf.placeholder(tf.float32, name='x_input')
    y_input = tf.placeholder(tf.float32, name='y_input')
    w = tf.Variable(2.0, name='weight')
    b = tf.Variable(1.0, name='biases')
    y = tf.add(tf.multiply(x_input, w), b)
    loss_op = tf.reduce_sum(tf.pow(y_input - y, 2)) / (2 * 32)
    train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss_op)

    '''tensorboard'''
    gradients_node = tf.gradients(loss_op, w)
    # print(gradients_node)
    # tf.summary.scalar('norm_grads', gradients_node)
    # tf.summary.histogram('norm_grads', gradients_node)
    # merged = tf.summary.merge_all()
    # writer = tf.summary.FileWriter('log')

    init = tf.global_variables_initializer()
    sess.run(init)

    '''构造数据集'''
    num = 32
    x_pure = np.random.randint(-10, 100, num)
    x_train = x_pure + np.random.randn(num) / 10  # 为x加噪声
    y_train = 3 * x_pure + 2 + np.random.randn(num) / 10  # 为y加噪声

    for i in range(32):
        _, gradients, loss = sess.run([train_op, gradients_node, loss_op],
                                    feed_dict={x_input: x_train[i], y_input: y_train[i]})
        print("epoch: {} \t loss: {} \t gradients: {}".format(i, loss, gradients), ", x: ", x_train[i])

    sess.close()

def GradientTest2():
    w1 = tf.get_variable('w1', shape=[3])
    w2 = tf.get_variable('w2', shape=[3])

    w3 = tf.get_variable('w3', shape=[3])
    w4 = tf.get_variable('w4', shape=[3])

    z1 = 3 * w1 + 2 * w2+ w3
    z2 = -1 * w3 + w4

    #grads = tf.gradients([z1, z2], [w1, w2, w3, w4])
    grads = tf.gradients([z1, z2], [w1, w2, w3, w4], grad_ys=[[-2.0, -3.0, -4.0], [-1.0, -2.0, -3.0]])

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        print(sess.run(grads))

#TfConvTest()
GradientTest2()

