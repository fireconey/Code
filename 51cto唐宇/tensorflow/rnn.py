import numpy as np
import tensorflow as tf
import matplotlib.pylab as plt
from tensorflow.contrib import rnn
import tensorflow.时间序列 as data_loader


class SereisPredictor():
    def __init__(self,input_dim,seq_size,hidden_dim):
        self.input_dim=input_dim
        self.seq_size=seq_size
        self.hidden_dim=hidden_dim

        self.w_out=tf.Variable(tf.random_normal([hidden_dim,1]),name="w_out")
        self.b_out=tf.Variable(tf.random_normal([1]),name="b_out")
        self.x=tf.placeholder(tf.float32,[None,seq_size,input_dim])
        self.y=tf.placeholder(tf.float32,[None,seq_size])

        self.cost=tf.reduce_mean(tf.square(self.model())-self.y)
        self.train_op=tf.train.AdamOptimizer(learing_rate=0.01).minnimize(self.cost)

        self.saver=tf.train.Saver()

    def model(self):
        cell=rnn.BasicLSTMCell(self.hidden_dim)
        outputs,state=tf.nn.dynamic_rnn(cell,self.x,self.y,dtype=tf.float32)
        num_examples=tf.shape(self.x)[0]
        w_repeated=tf.tile(tf.expand_dims(self.w_out,0),[num_examples,1,1])
        out=tf.matmul(outputs,w_repeated)+self.b_out
        return  out

    def train(self,train_x,train_y,test_x,test_y):
        with tf.Session() as sess:
            tf.get_variable_scope().reuse_variables()
            sess.run(tf.global_variables_initializer())
            max_patience=3
            patience=max_patience
            min_test_err=float("inf")
            step=0
            while patience>0:
                _,train_err=sess.run([self.train_op,self.cost],feed_dict={self.x:train_x,self.y:train_y})
                if step%100==0:
                    test_err=sess.run(self.cost,feed_dict={self.x:test_x,self.y:test_y})
                    print("step{}/训练错误{}测试错误{}".format(step,train_err,test_err))
                    if test_err < min_test_err:
                        min_test_err=test_err
                        patience=max_patience
                    else:
                        patience-=1
                step+=1
            save_path=self.saver.save(sess,"./model/")
            print("model保存{}".format(save_path))

    def test(self,sess,test_x):
        tf.get_variable_scope().reuse_variables()
        self.saver.restore(sess,"./model/")
        output=sess.run(self.model(),feed_dict={self.x:test_x})
        return  output


def plot_result(train_x,predictions,actual,filename):
    plt.figure()
    num_train=len(train_x)
    plt.plot(list(range(num_train)),train_x,color="b",label="训练数据")
    plt.plot(list(range(num_train,num_train+len(predictions))),predictions,color="r",label="预测值")
    plt.plot(list(range(num_train,num_train+len(actual))),actual,color="g",label="测试数据")
    plt.legend()
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()







if __name__ == '__main__':
    seq_size=5
    predector=SereisPredictor(input_dim=1,seq_size=seq_size,hidden_dim=100)
    data=data_loader.load_series("data.csv")
    train_data,actual_vals=data_loader.split_data(data)

    train_x,train_y=[],[]
    for i in range(len(train_data)-seq_size-1):
        train_x.append(np.expand_dims(train_data[i:i+seq_size],axis=1).tolist())
        train_y.append(train_data[i+1:i+seq_size+1])

    test_x,test_y=[],[]
    for i in range(len(actual_vals)-seq_size-1):
        test_x.append(np.expand_dims(actual_vals[i:i+seq_size],axis=1).tolist())
        test_y.append(actual_vals[i+1:i+seq_size+1])

    predector.train(train_x,train_y,test_x,test_y)

    with tf.Session() as sess:
        predict_vals=predector.test(sess,test_x)[:0]
        print("预测结果",np.shape(predict_vals))
        plot_result(train_data,predict_vals,actual_vals,"预测图.png")

        prev_seq=train_x[-1]
        predict_vals=[]
        for i in range(20):
            next_seq=predector.test(sess,[prev_seq])
            predict_vals.append(next_seq[-1])
            prev_seq=np.vstack(prev_seq[1:],next_seq[-1])
        plot_result(train_data,predict_vals,actual_vals,"检验结果.png")

