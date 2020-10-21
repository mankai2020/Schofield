import keras
import numpy as np
import matplotlib.pyplot as plt  
from keras.datasets import mnist  
from keras.models import Model  
from keras.layers import Input, add  
from keras.layers import Layer, Dense, Dropout, Activation, Flatten, Reshape  
from keras import regularizers  
from keras.regularizers import l2  
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D  
from keras.utils import np_utils
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

#(X_train, _), (X_test, _) = mnist.load_data()  
  
#  归一化  
#X_train = X_train.astype("float32")/255.  
#X_test = X_test.astype("float32")/255.  
  
# print('X_train shape:', X_train.shape)  
# print(X_train.shape[0], 'train samples')  
# print(X_test.shape[0], 'test samples')

# np.prod是将28X28矩阵转化成1X784,方便全连接神经网络输入层784个神经元读取。
#X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))  
#X_test = X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))

X_train = np.array([[149.5,69.5,38.5],[162.5,77.0,55.5],[162.7,78.5,50.8]\
	,[162.2,87.5,65.5],[156.5,74.5,49.0],[156.1,74.5,45.5],[172.0,76.5,51.0]\
	,[173.2,81.5,59.5],[159.5,74.5,43.5],[157.7,79.0,53.5]])
# X_train = np.array([[149.5,162.5,162.7,162.2,156.5,156.1,172.0,173.2,159.5,157.7],\
# 	[69.5,77.0,78.5,87.5,74.5,74.5,76.5,81.5,74.5,79.0],\
# 	[38.5,55.5,50.8,65.5,49.0,45.5,51.0,59.5,43.5,53.5]])
X_test = X_train

print(X_train)
print(len(X_train[0]))

input_size = 3 
hidden_size = 3 
output_size = 3  
    
x = Input(shape=(input_size,))  
h = Dense(hidden_size, activation='relu')(x)  
r = Dense(output_size, activation='sigmoid')(h)  
   
autoencoder = Model(inputs=x, outputs=r)  
autoencoder.compile(optimizer='adam', loss='mse')

epochs = 5  
batch_size = 128  
  
history = autoencoder.fit(X_train, X_train, batch_size=batch_size,epochs=epochs, verbose=1,validation_data=(X_test, X_test))

conv_encoder = Model(x, h)  # 只取编码器做模型  
encoded_imgs = conv_encoder.predict(X_test)  
print(encoded_imgs)

decoded_imgs = autoencoder.predict(X_test) 
print(decoded_imgs)
  
# # 打印10张测试集手写体的压缩效果  
# n = 10  
# plt.figure(figsize=(20, 8))  
# for i in range(n):  
# 	ax = plt.subplot(1, n, i+1)  
# 	plt.imshow(encoded_imgs[i].reshape(4, 16).T)  
# 	plt.gray()
# 	ax.get_xaxis().set_visible(False)  
# 	ax.get_yaxis().set_visible(False)  
# plt.show()

# decoded_imgs = autoencoder.predict(X_test)  
# n = 10  
# plt.figure(figsize=(20, 6))  
# for i in range(n):  
#     # 打印原图  
# 	ax = plt.subplot(3, n, i+1)  
# 	plt.imshow(X_test[i].reshape(28, 28))  
# 	plt.gray()  
# 	ax.get_xaxis().set_visible(False)  
# 	ax.get_yaxis().set_visible(False)  
   
      
#     # 打印解码图  
# 	ax = plt.subplot(3, n, i+n+1)  
# 	plt.imshow(decoded_imgs[i].reshape(28, 28))  
# 	plt.gray()  
# 	ax.get_xaxis().set_visible(False)  
# 	ax.get_yaxis().set_visible(False)  
       
# plt.show()