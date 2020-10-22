import numpy as np
import matplotlib as mpl 
import matplotlib.pyplot as plt 


def zeroMean(dataMat):
	meanValue = np.mean(dataMat, axis = 0)
	newData = dataMat - meanValue
	return newData,meanValue

def calN(eigVals,percentage):
	sortArray = (np.sort(eigVals))[-1::-1]
	arrSum = sum(sortArray)
	tmpSum = 0
	num = 0
	for i in sortArray:
		tmpSum += i
		num += 1
		if tmpSum >= percentage * arrSum:
			return num

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

dataMat = np.array([[2.5,2.4],[0.5,0.7],[2.2,2.9],[1.9,2.2],\
	[3.1,3.0],[2.3,2.7],[2.0,1.6],[1.0,1.1],[1.5,1.6],[1.1,0.9]])
lowDDataMat,reconDataMat = pca(dataMat,1)
print(dataMat)
print(lowDDataMat)
print(reconDataMat)
plt.scatter(reconDataMat[:,0].tolist(),reconDataMat[:,1].tolist(), marker='o')
plt.scatter(dataMat[:,0].tolist(),dataMat[:,1].tolist(), marker='^')
plt.show()