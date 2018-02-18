from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("data/",one_hot=True)
print(mnist.train.num_examples)
print(mnist.test.num_examples)


"""
trainimg=mnist.train.images
trainlabel=mnist.train.labels
"""