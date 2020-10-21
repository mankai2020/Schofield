import tensorflow as tf
import matplotlib.pyplot as pyplot
import numpy as numpy
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("MNIST_data/",one_hot=False)

learn_rate = 0.01
training_epochs = 20
batch_size = 1			#一个训练batch的数目
display_step = 1
example_to_show = 10
n_input = 784			#28*28
X=tf.placeholder("float",[None,n_input])	#参数，占位符，不定行，n_input

n_hidden_1 = 256		#隐藏层1维数
n_hidden_2 = 128		#隐藏层2维数

weights = {
	'encoder_w1':tf.Variable(tf.truncated_normal([n_input,n_hidden_1],)),
	'encoder_w2':tf.Variable(tf.truncated_normal([n_hidden_1,n_hidden_2],)),
	'encoder_w1':tf.Variable(tf.truncated_normal([n_hidden_2,n_hidden_1],)),
	'encoder_w2':tf.Variable(tf.truncated_normal([n_hidden_1,n_input],)),
}
biases={
	'encoder_b1':tf.Variable(tf.random_normal([n_hidden_1])),
	'encoder_b2':tf.Variable(tf.random_normal([n_hidden_2])),
	'encoder_b1':tf.Variable(tf.random_normal([n_hidden_1])),
	'encoder_b2':tf.Variable(tf.random_normal([n_input])),
}

def encoder(x):
	layer1 = tf.nn.sigmoid(tf.add(tf.matmul(x,weight['encoder_w1']),biases['encoder_b1']))
	layer2 = tf.nn.sigmoid(tf.add(tf.matmul(layer1,weight['encoder_w2']),biases['encoder_b2']))
	return layer2
#编码，\sigma(Wx+b)

def decoder(x):
	layer1 = tf.nn.sigmoid(tf.add(tf.matmul(x,weight['decoder_w1']),biases['decoder_b1']))
	layer2 = tf.nn.sigmoid(tf.add(tf.matmul(layer1,weight['decoder_w2']),biases['decoder_b2']))
	return layer2
#解码，\sigma(Wx+b)

with tf.Session() as sess:
	init = tf.global_variables_initializer()	#初始化
	sess.run(init)
	total_batch = int(mnist.train.num_examples/batch_size)	#batch数
	for epoch in range(training_epochs):
		for i in range(total_batch):
			batch_xs,batch_ys = mnist.train.next_batch(batch_size)	#随机选妻batch_size
			_,c = sess.run([optimizer,cost],feed_dict = {X:batch_xs})	#真实的数据代替占位符
		if epoch % display_step==0:
			print("Epoch:",'%04d' %(epoch+1),'cost=',"{:,.9f}".format(c))
	print("Optimization Finished")

	encode_decode = sess.run(y_predict,feed_dict = {X:mnist.test.images[:example_to_show]})
	f,a=plt.subplots(2,10,figsize=(10,2))
	for i in range(example_to_show):
		a[0][i].imshow(np.reshape(mnist.test.images[i],(28,28)))
		a[0][i].imshow(np.reshape(encode_decode[i],(28,28)))
	plt.show()