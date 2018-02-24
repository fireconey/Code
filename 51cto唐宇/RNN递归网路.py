#注意pycharm的初始目录是工程的更目录不是现对目录


#由于变量太多，有可能变量重复导致bug
#所以tf使用变量作用域来管理变量
#tf.get_variable(name,shape,init)通过名称创建或获取变量
#tf.variable_scope(name)给get_variable 设定名称空间
#初始化比较特殊，有自己的方法
#如果同意个变量赋值两次就会报错提醒你你的变量有变动，只有设定共享了，才可以
#使用variable_scope("ff")来指定共享区,
import tensorflow as tf
import  numpy as np
from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import  input_data

mnist=input_data.read_data_sets("tensorflow/data/",one_hot=True)

sess=tf.Session()

lr=1e-3               #训练率
input_size=28         #由于图片是28列28行，所以一条是28个数据
timestep_size=28      #有28条
hidden_size=256       #隐藏层的数量
layer_num=2           #堆叠层
class_num=10          #分类的数量


#由于读入数据的时候使用了hot_one所以是784个列。None是批次
_x=tf.placeholder(tf.float32,[None,784])
y=tf.placeholder(tf.float32,[None,class_num])

#获取的批次
batch_size=tf.placeholder(tf.int32,[])

#保留率
keep_prob=tf.placeholder(tf.float32,[])

#_x是读入的一定批次的一维的784个特征的数据，所以
#要变成28x28的数据
x=tf.reshape(_x,[-1,28,28])


#定义递归神经网络单元
def lstm_cell():
    #lstm:long short term memory
    #后面的reuse表示是否可以共享参数等号后面的是使其设为true
    #隐藏层是输入到输出之间的神经元
    #堆叠是同种神经网络的输出又输入到另一个的神经网络的输入中
    cell=rnn.LSTMCell(hidden_size,reuse=tf.get_variable_scope().reuse)

    #指定保留率
    return rnn.DropoutWrapper(cell,output_keep_prob=keep_prob)

    #堆叠层
mlstm_cell=rnn.MultiRNNCell([lstm_cell() for _ in range(layer_num)],state_is_tuple=True)

#对变量进行初始化，这里是同一个批次的所有图片的相同行同时计算，所以要指定批次大小
init_state=mlstm_cell.zero_state(batch_size,dtype=tf.float32)

output=list()
state=init_state

#在RNN的作用域下运行各种变量，这样，这种变量只有在做代码里有效
with tf.variable_scope("RNN"):
    #导入图片每一行的数据
    for timestep in range(timestep_size):
        if timestep>0:
            #每次代码可以重用的设置
            tf.get_variable_scope().reuse_variables()
            #mlstm_cell是上面定义的堆叠层
            #x是传入的参数，第一个：表示一个批次所以的图片
            #timstep是图片的行数，最后的是图片所有的列数
            #state是同一批次的所有接受器
        cell_output,state=mlstm_cell(x[:,timestep,:],state)
        #每个数据是所有图片相同行的数据
        output.append(cell_output)
# 得到最后的结果,由于这里取了一次值所以只有2个维度了，所以w可以是两个维度
h_sate=output[-1]

# 使用softmax进行归一化
w=tf.Variable(tf.truncated_normal([hidden_size,class_num],stddev=0.1,dtype=tf.float32))
bias=tf.Variable(tf.constant(0.1,shape=[class_num]),dtype=tf.float32)
y_preb=tf.nn.softmax(tf.matmul(h_sate,w)+bias)


#使用交叉熵来指导函数的训练的结果的好坏
cross_entropy=-tf.reduce_mean(y*tf.log(y_preb))
#指定训练步伐，进行训练
train_op=tf.train.AdadeltaOptimizer(lr).minimize(cross_entropy)

#找到预测正确的数据，返回的是布尔矩阵
correct_prediction=tf.equal(tf.argmax(y_preb,1),tf.argmax(y,1))
#通过均值计算正确率
accuray=tf.reduce_mean(tf.cast(correct_prediction,"float"))

sess.run(tf.global_variables_initializer())
for i in range(2000):
    _batch_size=128
    #本来获取的有训练集，训练集标签，测试集，测试集标签四个变量，但是
    #只使用一个变量来接收了(返回变成一个元祖），所以使用batch[]来指定了。
    batch=mnist.train.next_batch(_batch_size)
    if(i+1)%200==0:
        train_acc=sess.run(accuray,feed_dict={
            _x:batch[0],y:batch[1],keep_prob:1.0,
            batch_size:_batch_size
        })
        print("迭代%d,步奏%d,训练正确率%g" %(mnist.train.epochs_completed,(i+1),train_acc))

    sess.run(train_op,feed_dict={_x:batch[0],y:batch[1],keep_prob:0.5
                                     ,batch_size:_batch_size})

print("测试的准确率  %g" %sess.run(accuray,feed_dict={
    _x:mnist.test.images,y:mnist.test.labels,keep_prob:1.0,batch_size:mnist.test.images.shape[0]
}))
