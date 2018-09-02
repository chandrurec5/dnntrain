import numpy as np
import relu
# Implements a simple neural network#
# Logistic Loss, SGD are hardcoded#
# All layers have the same width, except for the last layer
class nn:
	bias_const=0.01
	def __init__(self,width,depth,num_class,temp): #temp is temperature
		self.width=width
		self.depth=depth
		self.num_class=num_class
		self.weight=np.zeros((width,width, depth),dtype=float)
		self.bias=np.zeros((width,depth),dtype=float)
		self.output=np.zeros((width,depth),dtype=float)
		self.active=np.zeros((width,depth),dtype=float)
		self.temp=temp
                self.delta_weight=np.zeros(np.shape(self.weight),dtype=float)
                self.delta_bias=np.zeros(np.shape(self.bias),dtype=float)
		self.logit=np.zeros((width,1),dtype=float)

#		self.initialize()
	def initialize(self):
		self.weight=np.random.normal(0,1/np.sqrt(self.width),np.shape(self.weight))
		self.bias=bias_const*np.ones(np.shape(self.bias),dtype=float)
		self.weight[self.num_class:self.width,:,self.depth-1]=np.zeros((self.width-self.num_class,self.width,1),dtype=float)
		self.bias[self.num_class:self.width,self.depth-1]=np.inf((self.width-self.num_class,1),dtype=float)
		
	def fp(self,xin):#Forward Propagation
		o=xin
		for i in xrange(self.depth):
			o=np.dot(self.weight[:,:,i],o)
			if (i!=self.depth-1) :
				[o,a]=relu(o+self.bias[:,i])
			else:
				a=np.ones((self.width,1),dtype=float)
			self.output[:,i]=o
			self.active[:,i]=a
		self.logit=self.output[:,self]
		return self.logit
	def bp(self,xin,yin):#Backward Propagation: yin is in one-hot format
		o=self.fp(self,xin)
		p=np.exp(o)
		p=p/np.sum(p)
		delta=-p
		delta[yin]=delta[yin]+1.0
		delta_weight=np.zeros(np.shape(self.weight),dtype=float)
                delta_bias=np.zeros(np.shape(self.bias),dtype=float)
		delta_bias_last=delta
		delta_weight_last=np.tile(delta,self.width)*np.diag(self.output[:,self.depth-1])
		weight_prev=self.weight_last	
		for i in xrange (self.depth-1,-1,0):
			delta_bias[:,i]=np.transpose(weight_prev)*delta*self_active[:,i]
			delta_weight[:,i]=np.tile(delta_bias,self.width)*np.diag(self.output[:,i-1])
			weight_prev=self.weight[:,i]	
		i=0
		delta_bias[:,i]=np.transpose(weight_prev)*delta*self_active[:,i]
                delta_weight[:,i]=np.tile(delta_bias,self.width)*np.diag(xin)

			

		
