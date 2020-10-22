import numpy as np
import matplotlib as mpl 
import matplotlib.pyplot as plt 


dataMat = np.array([[2.5,2.4],[0.5,0.7],[2.2,2.9],[1.9,2.2],\
	[3.1,3.0],[2.3,2.7],[2.0,1.6],[1.0,1.1],[1.5,1.6],[1.1,0.9]])
U,sigma,VT = np.linalg.svd(dataMat,full_matrices=1,compute_uv=1)
