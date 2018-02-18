"""
定义计算图:计算图是没有计算的，只是画个图
"""
import tensorflow as tf

graph=tf.Graph()
with graph.as_default():
	foot=tf.Variable(3,name="foo")
	bar=tf.Variable(2,name="bar")
	result=foot+bar
	initiable=tf.global_variables_initializer()
print(result)



"""
session才是真正的计算
"""
with tf.Session(graph=graph) as sess:
	sess.run(initiable)
	res=sess.run(result)
print(res)



"""
#rank表示数据的维度
#1、有scalar(常数)
#2、向量
#3、矩阵
#4、3、4、5等的多维

shap表示维度的个数
只表示最外一层的的数组或元祖的维度


data type表示单个数据的类型。
有：DT_FLOAT;DT_INT(8-16)等等

"""



"""
tf.matmul(x,x)
matmul表示的是矩阵乘
"""


"""
variable 是一个变量用于保存不会销毁的数据
如w值，b值，由于训练就是要得到w值和b值
所以不能再计算过程中消失所以使用variable
"""



"""
sigmod()逻辑函数用于神经网络的激活
"""

