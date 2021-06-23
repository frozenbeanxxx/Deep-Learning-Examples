import tenserflow as tf 

#一个简单的tf.Session例子
# 建立一个graph.
a = tf.constant(5.0)
b = tf.constant(6.0)
c = a * b

# 将graph载入到一个会话session中
sess = tf.Session()

# 计算tensor `c`.
print(sess.run(c))#一个会话可能会占用一些资源，比如变量、队列和读取器(reader)。释放这些不再使用的资源非常重要。
#使用close()方法关闭会话，或者使用上下文管理器，释放资源。
# 使用`close()`方法.
sess = tf.Session()
sess.run(...)
sess.close()

# 使用上下文管理器
with tf.Session() as sess:
    sess.run(...)# Launch the graph in a session that allows soft device placement and
# logs the placement decisions.
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                        log_device_placement=True))
    a = tf.constant([10, 20])
    b = tf.constant([1.0, 2.0])
    # 'fetches' 可以为单个数
    v = session.run(a)
    # v is the numpy array [10, 20]
    # 'fetches' 可以为一个list.
    v = session.run([a, b])
    # v a Python list with 2 numpy arrays: the numpy array [10, 20] and the
    # 1-D array [1.0, 2.0]
    # 'fetches' 可以是 lists, tuples, namedtuple, dicts中的任意:
    MyData = collections.namedtuple('MyData', ['a', 'b'])
    v = session.run({'k1': MyData(a, b), 'k2': [b, a]})
    # v 为一个dict，并有
    # v['k1'] is a MyData namedtuple with 'a' the numpy array [10, 20] and
    # 'b' the numpy array [1.0, 2.0]
    # v['k2'] is a list with the numpy array [1.0, 2.0] and the numpy array
    # [10, 20].c = tf.constant(..)
sess = tf.Session()

with sess.as_default():
    assert tf.get_default_session() is sess
    print(c.eval())c = tf.constant(...)
sess = tf.Session()
with sess.as_default():
    print(c.eval())
# ...
with sess.as_default():
    print(c.eval())
#关闭会话
sess.close()
#使用 with tf.Session()方式可以创建并自动关闭会话sess = tf.InteractiveSession()
a = tf.constant(5.0)
b = tf.constant(6.0)
c = a * b
# 我们直接使用'c.eval()' 而没有通过'sess'
print(c.eval())
sess.close()a = tf.constant(5.0)
b = tf.constant(6.0)
c = a * b
with tf.Session():
    # We can also use 'c.eval()' here.
    print(c.eval())
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
global_step = tf.Variable(0, name='global_step', trainable=False)
train_op = optimizer.minimize(loss, global_step=global_step)# Create an optimizer with the desired parameters.
opt = GradientDescentOptimizer(learning_rate=0.1)
# Add Ops to the graph to minimize a cost by updating a list of variables.
# "cost" is a Tensor, and the list of variables contains tf.Variable objects.
opt_op = opt.minimize(cost, var_list=<list of variables>)
# Execute opt_op to do one step of training:
opt_op.run()
1、使用函数compute_gradients()计算梯度
2、按照自己的愿望处理梯度
3、使用函数apply_gradients()应用处理过后的梯度# 创建一个optimizer.
opt = GradientDescentOptimizer(learning_rate=0.1)

# 计算<list of variables>相关的梯度
grads_and_vars = opt.compute_gradients(loss, <list of variables>)

# grads_and_vars为tuples (gradient, variable)组成的列表。
#对梯度进行想要的处理，比如cap处理
capped_grads_and_vars = [(MyCapper(gv[0]), gv[1]) for gv in grads_and_vars]

# 令optimizer运用capped的梯度(gradients)
opt.apply_gradients(capped_grads_and_vars)m_0 <- 0 (Initialize initial 1st moment vector)
v_0 <- 0 (Initialize initial 2nd moment vector)
t <- 0 (Initialize timestep)t <- t + 1
lr_t <- learning_rate * sqrt(1 - beta2^t) / (1 - beta1^t)

m_t <- beta1 * m_{t-1} + (1 - beta1) * g
v_t <- beta2 * v_{t-1} + (1 - beta2) * g * g
variable <- variable - lr_t * m_t / (sqrt(v_t) + epsilon)#该函数返回以下结果
decayed_learning_rate = learning_rate *
         decay_rate ^ (global_step / decay_steps)
##例： 以0.96为基数，每100000 步进行一次学习率的衰退
global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 0.1
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           100000, 0.96, staircase=True)
# Passing global_step to minimize() will increment it at each step.
learning_step = (
    tf.train.GradientDescentOptimizer(learning_rate)
    .minimize(...my loss..., global_step=global_step)
)# Example usage when creating a training model:
# Create variables.
var0 = tf.Variable(...)
var1 = tf.Variable(...)
# ... use the variables to build a training model...
...
# Create an op that applies the optimizer. This is what we usually
# would use as a training op.
opt_op = opt.minimize(my_loss, [var0, var1])

# Create an ExponentialMovingAverage object
ema = tf.train.ExponentialMovingAverage(decay=0.9999)

# Create the shadow variables, and add ops to maintain moving averages
# of var0 and var1.
maintain_averages_op = ema.apply([var0, var1])

# Create an op that will update the moving averages after each training
# step. This is what we will use in place of the usual training op.
with tf.control_dependencies([opt_op]):
    training_op = tf.group(maintain_averages_op)

...train the model by running training_op...

#Example of restoring the shadow variable values:
# Create a Saver that loads variables from their saved shadow values.
shadow_var0_name = ema.average_name(var0)
shadow_var1_name = ema.average_name(var1)
saver = tf.train.Saver({shadow_var0_name: var0, shadow_var1_name: var1})
saver.restore(...checkpoint filename...)
# var0 and var1 now hold the moving average valuesvariables_to_restore = ema.variables_to_restore()
  saver = tf.train.Saver(variables_to_restore)#Coordinator的使用，用于多线程的协调
try:
  ...
  coord = Coordinator()
  # Start a number of threads, passing the coordinator to each of them.
  ...start thread 1...(coord, ...)
  ...start thread N...(coord, ...)
  # Wait for all the threads to terminate, give them 10s grace period
  coord.join(threads, stop_grace_period_secs=10)
except RuntimeException:
  ...one of the threads took more than 10s to stop after request_stop()
  ...was called.
except Exception:
  ...exception that was passed to coord.request_stop()with coord.stop_on_exception():
  # Any exception raised in the body of the with
  # clause is reported to the coordinator before terminating
  # the execution of the body.
  ...body...
#等价于
try:
  ...body...
exception Exception as ex:
  coord.request_stop(ex)server = tf.train.Server(...)
with tf.Session(server.target):
  # ...with tf.Graph().as_default():
  ...add operations to the graph...
  # Create a Supervisor that will checkpoint the model in '/tmp/mydir'.
  sv = Supervisor(logdir='/tmp/mydir')
  # Get a TensorFlow session managed by the supervisor.
  with sv.managed_session(FLAGS.master) as sess:
    # Use the session to train the graph.
    while not sv.should_stop():
      sess.run(<my_train_op>)
# 在上下文管理器with sv.managed_session()内，所有在graph的变量都被初始化。
# 或者说，一些服务器checkpoint相应模型并增加summaries至事件log中。
# 如果有例外发生，should_stop()将返回True# Choose a task as the chief. This could be based on server_def.task_index,
# or job_def.name, or job_def.tasks. It's entirely up to the end user.
# But there can be only one *chief*.
is_chief = (server_def.task_index == 0)
server = tf.train.Server(server_def)

with tf.Graph().as_default():
  ...add operations to the graph...
  # Create a Supervisor that uses log directory on a shared file system.
  # Indicate if you are the 'chief'
  sv = Supervisor(logdir='/shared_directory/...', is_chief=is_chief)
  # Get a Session in a TensorFlow server on the cluster.
  with sv.managed_session(server.target) as sess:
    # Use the session to train the graph.
    while not sv.should_stop():
      sess.run(<my_train_op>)#例如： 开启一个线程用于打印loss. 设置每60秒该线程运行一次，我们使用sv.loop()
 ...
  sv = Supervisor(logdir='/tmp/mydir')
  with sv.managed_session(FLAGS.master) as sess:
    sv.loop(60, print_loss, (sess))
    while not sv.should_stop():
      sess.run(my_train_op)在chief中每100个step，创建summaries
  # Create a Supervisor with no automatic summaries.
  sv = Supervisor(logdir='/tmp/mydir', is_chief=is_chief, summary_op=None)
  # As summary_op was None, managed_session() does not start the
  # summary thread.
  with sv.managed_session(FLAGS.master) as sess:
    for step in xrange(1000000):
      if sv.should_stop():
        break
      if is_chief and step % 100 == 0:
        # Create the summary every 100 chief steps.
        sv.summary_computed(sess, sess.run(my_summary_op))
      else:
        # Train normally
        sess.run(my_train_op)def train():
  sv = tf.train.Supervisor(...)
  with sv.managed_session(<master>) as sess:
    for step in xrange(..):
      if sv.should_stop():
        break
      sess.run(<my training op>)
      ...do other things needed at each training step...with tf.Graph().as_default():
   ...add operations to the graph...
  # Create a SessionManager that will checkpoint the model in '/tmp/mydir'.
  sm = SessionManager()
  sess = sm.prepare_session(master, init_op, saver, checkpoint_dir)
  # Use the session to train the graph.
  while True:
    sess.run(<my_train_op>)
#其中prepare_session()初始化和恢复一个模型参数。 

#另一个进程将等待model准备完成，代码如下
with tf.Graph().as_default():
  ...add operations to the graph...
  # Create a SessionManager that will wait for the model to become ready.
  sm = SessionManager()
  sess = sm.wait_for_session(master)
  # Use the session to train the graph.
  while True:
    sess.run(<my_train_op>)
#wait_for_session()等待一个model被其他进程初始化cluster = tf.train.ClusterSpec({"worker": ["worker0.example.com:2222",
                                           "worker1.example.com:2222",
                                           "worker2.example.com:2222"],
                                "ps": ["ps0.example.com:2222",
                                       "ps1.example.com:2222"]})# To build a cluster with two ps jobs on hosts ps0 and ps1, and 3 worker
# jobs on hosts worker0, worker1 and worker2.
cluster_spec = {
    "ps": ["ps0:2222", "ps1:2222"],
    "worker": ["worker0:2222", "worker1:2222", "worker2:2222"]}
with tf.device(tf.replica_device_setter(cluster=cluster_spec)):
  # Build your graph
  v1 = tf.Variable(...)  # assigned to /job:ps/task:0
  v2 = tf.Variable(...)  # assigned to /job:ps/task:1
  v3 = tf.Variable(...)  # assigned to /job:ps/task:0
# Run compute...create a graph...
# Launch the graph in a session.
sess = tf.Session()
# Create a summary writer, add the 'graph' to the event file.
writer = tf.train.SummaryWriter(<some-directory>, sess.graph)#打印时间文件中的内容
for e in tf.train.summary_iterator(path to events file):
    print(e)

#打印指定的summary值
# This example supposes that the events file contains summaries with a
# summary value tag 'loss'. These could have been added by calling
# `add_summary()`, passing the output of a scalar summary op created with
# with: `tf.scalar_summary(['loss'], loss_tensor)`.
for e in tf.train.summary_iterator(path to events file):
    for v in e.summary.value:
        if v.tag == 'loss':
            print(v.simple_value)# Creates a variable to hold the global_step.
global_step_tensor = tf.Variable(10, trainable=False, name='global_step')
# Creates a session.
sess = tf.Session()
# Initializes the variable.
sess.run(global_step_tensor.initializer)
print('global_step: %s' % tf.train.global_step(sess, global_step_tensor))

global_step: 10v = tf.Variable(0, name='my_variable')
sess = tf.Session()
tf.train.write_graph(sess.graph_def, '/tmp/my-model', 'train.pbtxt')#tf.py_func(func, inp, Tout, stateful=True, name=None)
#func：为一个python函数
#inp：为输入函数的参数，Tensor列表
#Tout： 指定func返回的输出的数据类型，是一个列表
def my_func(x):
  # x will be a numpy array with the contents of the placeholder below
  return np.sinh(x)
inp = tf.placeholder(tf.float32, [...])
y = py_func(my_func, [inp], [tf.float32])import tensorflow as tf


class SquareTest(tf.test.TestCase):

  def testSquare(self):
    with self.test_session():
      x = tf.square([2, 3])
      self.assertAllEqual(x.eval(), [4, 9])


if __name__ == '__main__':
  tf.test.main()