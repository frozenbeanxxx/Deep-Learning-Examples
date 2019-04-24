import tensorflow as tf 

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


TfConvTest()

