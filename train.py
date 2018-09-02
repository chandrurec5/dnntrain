import numpy as np




def main():
	file=open("data.csv","r")
	y=np.loadtxt("data.csv", delimiter=" ",unpack=False)	
	
if __name__=="__main__":
	main()
