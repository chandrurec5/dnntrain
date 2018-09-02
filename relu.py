import numpy as np
import matplotlib.mlab as mlab
def relu(x):
	y=np.zeros(np.size(x),dtype=float )
	z=np.maximum(x,np.zeros(np.size(x),dtype=float))
	idx=mlab.find(z>0.0)
	y[ind]=1.0
	return z,y 	
        
