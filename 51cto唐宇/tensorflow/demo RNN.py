import tensorflow as tf
from tensorflow.contrib import  rnn
class SeriesPredictor():
    def __init__(self,input_dim,seq_size,hidden_dim=10):
        self.input_dim=input_dim
        self.seq_size=seq_size
        self.hidden_dim=hidden_dim

        #由于隐藏层是hideen_dim所以输出有hidden_dim个特征
        #由于每次只导入一个神经元所以后面为1
        #由于有3个批次所以是3，由于w不是feed所以要指定为3不能指定None
        #这里我没有指定，后面进行扩维度
        self.w_out=tf.Variable(tf.random_normal([hidden_dim,1]),name="w")

        #x*w后对应的维度的数应相同，如图片为3*4*1所以b要3*4*1
        #如果是3*1中的1和图片的4就对不上所以报错。
        #同时要么b指定和其一样的维度，否则可以指定一个常数
        #如果[1]型
        self.b_out=tf.Variable(tf.random_normal([1]),name="b")

        #由于每次输入一行全列，但是有seq_size行，input_dim列
        self.x=tf.placeholder(tf.float32,[None,seq_size,input_dim])
        self.y=tf.placeholder(tf.float32,[None,seq_size])

        self.cost=tf.reduce_mean(tf.square(self.model()-self.y))
        self.train_op=tf.train.AdamOptimizer().minimize(self.cost)

        self.saver=tf.train.Saver()


    def model(self):
        #构建一定隐藏层的神经网络
        #神经元会更具行自动的切行，整体传入就行了
        cell=rnn.BasicLSTMCell(self.hidden_dim)
        output,stats=tf.nn.dynamic_rnn(cell,self.x,dtype=tf.float32)

        #shape得到形状数据[0]得到第一个数据
        #得到有多少批次
        num_examples=tf.shape(self.x)[0]
        print(num_examples,"9999999")
         #由于输出结果没有取所以输出结果是3个维度，所以w要扩维度
        #在没有传入数据的变量中只有这种方式，w不是这个情况也可以使用
        #reshape()
        tf_expend=tf.expand_dims(self.w_out,0)
        #tile表示在其维度内复制个数如果没有这个维度就不能复制
        #如[1]后面的参数是【2，2】第一个2表示在在第一维上复制
        #2个变成[1,1]，第二个2由于没有另一个维度所以报错
        #如果有另一个维度就变成[[1,1],[1,1]]
        tf_tile=tf.tile(tf_expend,[num_examples,1,1])
        out=tf.matmul(output,tf_tile)
        print(out,"********")
        out=out+self.b_out

        #如果有一个维度是一行数据，或一个列数据就减少这个维度
        #如二维的[[1,2]] 第一位就一个数据，可以不要这个维度了
        #变成[1,2]
        out=tf.squeeze(out)
        return  out

    def train(self,train_x,train_y):
        with tf.Session() as sess:
            tf.get_variable_scope().reuse_variables()
            sess.run(tf.global_variables_initializer())
            for i in range(10000):
                _,msg=sess.run([self.train_op,self.cost],feed_dict={self.x:train_x,self.y:train_y})
                if i%100==0:
                    print(i,msg)
            save_path=self.saver.save(sess,"model/predict")
            print("model 保存到{}".format(save_path))

    def test(self,test_x):
        with tf.Session() as sess:
            tf.get_variable_scope().reuse_variables()
            self.saver.restore(sess,"model/predict")
            output=sess.run(self.model(),feed_dict={self.x:test_x})
            return  output





if __name__=="__main__":
    #inupt_dim数据每次输入的列的数量，seq_size=4行

    predictor=SeriesPredictor(input_dim=1,seq_size=4,hidden_dim=10)
    #数据形状是3x4x1，是3批次,4行,1列
    train_x=[[[1],[2],[5],[6]],
             [[5],[7],[7],[8]],
             [[3],[4],[5],[7]]
            ]
    train_y=[[1,3,7,11],
             [5,12,14,15],
             [3,7,9,12]
             ]

    predictor.train(train_x,train_y)

    test_x=[
        [[1],[2],[3],[4]],
        [[4],[5],[6],[7]],
        [[3],[6],[7],[9]]
    ]

    acctual_y=[[1,3,5,7],
               [4,9,11,13],
               [3,9,13,16]
               ]
    print_y=predictor.test(test_x)
    print("开始测试")
    for i ,x in enumerate(test_x):
        print("当 输入是{}".format(x))
        print("实际值是{}".format(acctual_y[i]))
        print("预测值是{}".format(print_y[i]))
