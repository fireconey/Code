import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets as input_data
mnist=input_data("data/",one_hot=True)
x=tf.placeholder(tf.float32,[None,784])
y_=tf.placeholder(tf.float32,[None,10])
w=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10]))

y=tf.nn.softmax(tf.matmul(x,w)+b)



# 判断最优的函数交叉商
cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1]))

#进行训练
train_step=tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
init=tf.global_variables_initializer()
with tf.Session() as sess:
	result=sess.run(init)
	for i in range(1000):
		batch_xs,batch_ys=mnist.train.next_batch(100)
		sess.run(train_step,feed_dict={x:batch_xs,y_:batch_ys})
	correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    # cast是进行类型的转换
	accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
	print(sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels}))