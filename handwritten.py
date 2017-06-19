import pandas as pd
import numpy as np
import random
def load_data(filename):
    data=pd.read_csv(filename,sep=',')
    x=data.iloc[:,:-1].values
    y=data.iloc[:,-1].values
    return  x,y

class neuralNetwrok:
    
    def __init__(self,size):
        self.layers=len(size)
        self.size=size
        self.b=np.array([np.random.randn(i,1) for i in size[1:]])
        self.weight=np.array([np.random.randn(i,j) for i,j in zip(size[1:],size[:-1])])
        print " dataset loaded"

    
   def split_dataset(self,x,y,testsize):
        #fuction to split dataset 
        from sklearn.cross_validation import train_test_split
        x_train,x_test,y_train,y_test=train_test_split(x,y,testsize=testsize)
        print 'dataset splitted'
   
    # gradient fuction divides the data set into mini batches and apply gradient descent to all the minibatches 

    def gradientDescent(self,batch_size,train_data,time,rate):
        # train_data = (x,y) (X_train,y_train)
        for i in range(time):
            #times traing will be done 
            random.shuffle(train_data)
            minibatch=[train_data[k:k+batch_size] for k in xrange(0,len(train_data),batch_size)]
            print " minibatch created"
            for min_batch in minibatch:
                self.runGD(min_batch,rate)
    

    def runGD(self,min_batch,rate):
        # to run gradientDescent on a minibatch  
        B_gradient=[np.zeros(b.shape) for b in self.b)]
        W_gradient=[np.zeros(w.shape) for w in self.w)]
        for data in min_batch:
            b_gradient, w_gradient = backprob(data)
            B_gradient =[ x+y for x,y in zip(B_gradient,b_gradient)]
            W_gradient=[ x+y for x,y in zip(W_gradient,w_gradient)]
        # updating the parameters 
        self.b=[x-(rate/len(min_batch))*y for x,y in zip(self.b,B_gradient)]
        self.b=[x-(rate/len(min_batch))*y for x,y in zip(self.weight,W_gradient)]
    

    def backprob(data):
        #apply backprob with data on the neural network
        """ steps of backprob : 1. forward propagation to evalute the model 
                                2. apply backprob """
        


   


n=neuralNetwrok([2,3,1])
t=[[1,2],[2,3],[2,2],[3,7],[23,23],[23,232]]
n.gradientDescent(3,t)

        
