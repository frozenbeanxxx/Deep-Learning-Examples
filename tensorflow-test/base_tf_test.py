import tensorflow as tf 
import tensorflow.contrib.slim as slim
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

def MatMulTest():
    x = tf.constant([[[1.0,2.0,3.0],\
                    [1.0,2.0,3.0]]])
    y = tf.constant([[[3.0,4.0,5.0],\
                    [4.0,5.0,6.0]]])
    print(x)
    print(y)
    print(x*y)
    init_op = tf.initialize_all_variables()
    '''
    手机不支持矩阵转置乘法，这个地方直接换成点乘
    tf.matmul(x,y,transpose_b=True) as long as the second argument is constant and transposition is not used
    参考https://www.tensorflow.org/lite/guide/ops_compatibility
    '''
    t = tf.matmul(x, y, transpose_b=True)
    t = tf.squeeze(t, axis=[0])
    #t2 = slim.bias_add(t)
    with tf.Session() as sess:
        sess.run(init_op)
        print(sess.run(x*y))
        print(sess.run(t))
        #print(sess.run(t2))

def ConcatTest():
    x0 = tf.constant([[1.0,2.0,3.0],\
                    [1.0,2.0,3.0]])
    y0 = tf.constant([[3.0,4.0,5.0],\
                    [4.0,5.0,6.0]])
    x = tf.constant([[1,2,3],\
                    [1,2,3]])
    y = tf.constant([[3,4,5],\
                    [4,5,6]])
    print(x)
    print(y)
    r = tf.concat([x, y], 0)
    r1 = tf.concat([x0, y0], 1)
    r2 = tf.concat([x0, y0], -1)
    with tf.Session() as sess:
        print(sess.run(r))
        print(sess.run(r1))
        print(sess.run(r2))

def TfBaseAPI():
    a = tf.placeholder(tf.float32)
    b = tf.placeholder(tf.float32)
    y = tf.math.multiply(a, b)
    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter('./log',sess.graph)
        for step in range(10):  
            yy = sess.run(y, feed_dict={a:3, b:4})
            train_summary = tf.Summary(value=[tf.Summary.Value(tag="yy", simple_value=yy)])
            print(yy, end='\t')
            train_writer.add_summary(train_summary,step)
        train_writer.close()
        print()
    # tensorboard --logdir=log

    a = tf.constant([[1,2,3],\
                     [2,3,4]])
    b = tf.constant([[3,4,5],\
                     [4,5,6]])
    c = tf.constant([[3.1,4.2,5.5],\
                     [4.7,5.8,6.9]])
    y1 = tf.add(a, b, name='add')
    print(y1)
    print(y1.graph)
    with tf.Graph().as_default() as g:
        with g.name_scope("nested") as scope:
            y2 = tf.subtract(a, b, name='subtract')
            print(y2)
            print(y2.name)
            print(y2.graph)
    y3 = tf.abs(y2, name='abs')
    y4 = tf.multiply(a, b, name='multiply')
    y5 = tf.div(a, b, name='div')
    y6 = tf.divide(a, b, name='divide')
    y7 = tf.mod(a, b, name='mod')
    y8 = tf.negative(a, name='negative')
    y9 = tf.sign(a, name='sign') # y=sign(x) y = -1 if x<0; 0 if x==0; 1 if x>0
    y10 = tf.reciprocal(c, name='inv') # in latest release, reciprocal replace inv
    y11 = tf.square(a, name='square')
    y12 = tf.round(c)
    y13 = tf.sqrt(c)
    y14 = tf.pow(a, 2)
    y15 = tf.pow(a, b)
    y16 = tf.exp(c)
    y17 = tf.log(c)
    y18 = tf.maximum(a, b)
    y19 = tf.minimum(a, b)
    y20 = tf.cos(c)
    y21 = tf.sin(c)
    y22 = tf.tan(c)
    y23 = tf.atan(c)

    with tf.Session() as sess:
        print(sess.run(y1))
        print(sess.run(y2))
        print(sess.run(y3))
        print(sess.run(y4))
        print(sess.run(y5))
        print(sess.run(y6))
        print(sess.run(y7))
        print(sess.run(y8))
        print(sess.run(y9))
        print(sess.run(y10))
        print(sess.run(y11))
        print(sess.run(y12))
        print(sess.run(y13))
        print(sess.run(y14))
        print(sess.run(y15))
        print(sess.run(y16))
        print(sess.run(y17))
        print(sess.run(y18))
        print(sess.run(y19))
        print(sess.run(y20))
        print(sess.run(y21))
        print(sess.run(y22))
        print(sess.run(y23))

def TensorTransformationTest():
    s = tf.constant('1234567890')
    print('s: ', s)
    y1 = tf.string_to_number(s)
    print('y1: ', y1)
    y2 = tf.to_int32(y1)
    print('y2: ', y2)
    y3 = tf.to_int64(y1)
    print('y3: ', y3)
    y4 = tf.to_float(y1)
    print('y4: ', y4)
    y5 = tf.to_double(y1)
    print('y5: ', y5)
    y6 = tf.cast(y1, tf.int8)
    print('y6: ', y6)

    with tf.Session() as sess:
        print(sess.run(y1))
        print(sess.run(y2))
        print(sess.run(y3))
        print(sess.run(y4))
        print(sess.run(y5))
        print(sess.run(y6))

    m = tf.constant([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]],[[5, 5, 5], [6, 6, 6]]])
    m2 = tf.constant([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]],[[5, 5, 5], [6, 6, 6]]])
    t1 = [[1, 2, 3], [4, 5, 6]]
    t2 = [[7, 8, 9], [10, 11, 12]]
    z1 = tf.shape(m)
    z2 = tf.size(m)
    z3 = tf.rank(m) # rank means dimension
    z4 = tf.reshape(m, [2,-1])
    z5 = tf.expand_dims(m, 0)
    z6 = tf.slice(m, [1,0,0], [1,2,2])
    z71, z72, z73 = tf.split(m, 3, 0)
    z8 = tf.concat([t1,t2], 1)
    z9 = tf.stack([m,m2], axis=3) # tf.pack change to tf.stack
    z101 = tf.reverse(m, axis=[0,1])
    z102 = tf.reverse(m, axis=[1,0])
    z11 = tf.transpose(m, perm=[0,2,1])
    z12 = tf.gather(m, [0,2], axis=2)
    indices = tf.constant([1,2,4,5])
    z13 = tf.one_hot(indices, 6, 1.0, 0.0, axis=-1)
    with tf.Session() as sess:
        print(sess.run(z1))
        print(sess.run(z2))
        print(sess.run(z3))
        print(sess.run(z4))
        print(sess.run(z5))
        print(sess.run(z6))
        print('z71', sess.run(z71))
        print('z8', sess.run(z8))
        print('z9', sess.run(z9))
        print('z101', sess.run(z101))
        print('z102', sess.run(z102))
        print('z11', sess.run(z11))
        print('z12', sess.run(z12))
        print('z13', sess.run(z13))

def MatOperator():
    diagonal = [1,2,3,4]
    t1 = tf.diag(diagonal)
    t2 = tf.diag_part(t1)
    t3 = tf.trace(t1)
    n = np.array([i for i in range(9)]).reshape([3,-1])
    print(n)
    t4 = tf.matmul(n, n)
    m = tf.constant([[1.0,2.0,3.0], [2.0,3.0,6.0], [4.0,5.0,9.0]])
    t5 = tf.matrix_determinant(m)
    t6 = tf.matrix_inverse(m)
    #t7 = tf.cholesky(m)
    I = tf.eye(3)
    t8 = tf.matrix_solve(m, I) # 求解线性方程组 m*t8=I

    # complex
    t9 = tf.complex([1.2, 3.0], [2.3, 4.0])
    print('t9: ',t9, type(t9))
    t10 = tf.abs(t9)
    t11 = tf.conj(t9)
    t12 = tf.imag(t9)
    t13 = tf.real(t9)
    t14 = tf.fft(t9)

    with tf.Session() as sess:
        print(sess.run(t1))
        print(sess.run(t2))
        print(sess.run(t3))
        print(sess.run(t4))
        print(sess.run(t5))
        print(sess.run(t6))
        #print(sess.run(t7))
        print(sess.run(t8))
        print(sess.run(t9))
        print(sess.run(t10))
        print(sess.run(t11))
        print(sess.run(t12))
        print(sess.run(t13))
        print(sess.run(t14))

def ReductionTest():# 归并运算
    m = tf.constant([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]],[[5, 5, 5], [6, 6, 6]]], dtype=tf.float32)
    r1 = tf.reduce_sum(m)
    r2 = tf.reduce_prod(m)
    r3 = tf.reduce_min(m)
    r4 = tf.reduce_max(m)
    r5 = tf.reduce_mean(m)
    m2 = m > 3.5
    r6 = tf.reduce_all(m2)
    r7 = tf.reduce_any(m2)
    r8 = tf.accumulate_n([m,m])
    r9 = tf.cumsum([m,m,m])
    with tf.Session() as sess:
        print(sess.run(m))
        print(sess.run(r1))
        print(sess.run(r2))
        print(sess.run(r3))
        print(sess.run(r4))
        print(sess.run(r5))
        print(sess.run(m2))
        print(sess.run(r6))
        print(sess.run(r7))
        print(sess.run(r8))
        print(sess.run(r9))

#TfConvTest()
#GradientTest2()
#MatMulTest()
#ConcatTest()
#TfBaseAPI()
#TensorTransformationTest()
#MatOperator()
ReductionTest()

