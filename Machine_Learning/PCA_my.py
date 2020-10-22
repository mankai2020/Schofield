import numpy as np
import matplotlib as mpl 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import csv


def zeroMean(dataMat):
	meanValue = np.mean(dataMat, axis = 0)
	newData = dataMat - meanValue
	return newData,meanValue

def pca(dataMat,n):
	newData,meanValue = zeroMean(dataMat)
	covMat = np.cov(newData, rowvar = 0)
	eigVals,eigVects = np.linalg.eig(np.mat(covMat))
	eigValIndice = np.argsort(eigVals)
	n_eigValIndice = eigValIndice[-1:-(n+1):-1]
	n_eigVect = eigVects[:,n_eigValIndice]
	lowDDataMat = newData*n_eigVect
	reconDataMat = (lowDDataMat*n_eigVect.T) + meanValue
	return lowDDataMat,reconDataMat

#读取数据
p = r'/Users/mankai/Documents/满凯的资料/课程资料/机器学习基础/作业/data.csv'
with open(p,encoding = 'utf-8') as f:
    dataMat = np.loadtxt(f,delimiter = ",",skiprows = 1)

#执行PCA过程
lowDDataMat,reconDataMat = pca(dataMat,2)
print("原数据为：\n",dataMat)
print("降维之后的数据为\n",reconDataMat)

#降2维后的图表展示结果
# plt.scatter(reconDataMat[:,0].tolist(),reconDataMat[:,1].tolist(), marker='o')
# plt.scatter(dataMat[:,0].tolist(),dataMat[:,1].tolist(), marker='^')
# plt.show()

#降1维后的图表展示结果
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(reconDataMat[:,0].tolist(),reconDataMat[:,1].tolist(),reconDataMat[:,2].tolist(),marker='o')
ax.scatter(dataMat[:,0].tolist(),dataMat[:,1].tolist(),dataMat[:,2].tolist(),marker='^')
 
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
 
plt.show()