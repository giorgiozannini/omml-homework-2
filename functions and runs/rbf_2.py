# file imports
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from scipy.optimize import minimize as minimize
import time
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# splitting data in train test and val set
def data_split(data, val = True):
    
    random.seed(1696991)
    X = np.array(data.iloc[:,:2])
    y = np.array(data.iloc[:, 2])
    
    # train-val split 100% -> 70% - 30%
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=1) 
    
    if val == False:
        return X_train.T, X_val.T, y_train, y_val
    
    else:
    # val-test split 30% -> 15% - 15%
        X_test, X_val, y_test, y_val = train_test_split(X_val, y_val, test_size=0.5, random_state=1)
        return X_train.T, X_test.T, X_val.T, y_train, y_test, y_val
    
# common class for shallow neural newtworks with 1 output node
class ShallowNeuralNetwork:
    
    
    
    # defines the variables for a shallow nn with scalar output
    def __init__(self, X, y, N, sigma, rho, method = None, seed = 1):
        


        self.X = X
        self.y = y
        self.N = N
        np.random.seed(seed)
        self.c = self.X.T[np.random.randint(self.X.shape[1], size=self.N),:]
        self.v = np.random.normal(0,1,(1, self.N))
        self.output = np.zeros(y.shape[0])
        self.rho = rho
        self.sigma = sigma
        self.method = method
    
    # objective function to minimize
    def loss(self,params):
        
        v = params
        return 0.5 * np.mean(np.square((self.predict(self.X, self.c, v) - self.y))) +\
            self.rho*np.square(np.linalg.norm(np.concatenate([array.reshape(-1) for array in [self.c, v]])))
    
    # metric for train and test accuracy
    def mse(self, X, y, c, v):
        return 0.5 * np.mean(np.square(self.predict(X, c, v) - y)) 

# Mlp inherits the general charectiristics of a shallow nn 
class Rbf_el(ShallowNeuralNetwork):
    
    
    
    
    # activation function (hyperbolic tangent)
    def phi(self, x, c):
        
        x_c = np.stack([x.T]*self.N) - np.split(c, self.N)
        phi = np.exp(-np.square(np.array([np.linalg.norm(arr, axis = 1) for arr in (x_c)])/self.sigma))

        return phi
    
    def predict(self, x, c, v):

        H = self.phi(x, c)
        self.output = v @ H
        return self.output
    
    def grad(self, v):
        
        H = self.phi(self.X,self.c)
        dv = 2*self.rho*v + (1/self.X.shape[1]) * (H @ ((H.T @ v) - self.y).T )
        
        return dv
                          
    # optimization of the objective function
    def optimize(self):
        
        inits = self.v
        
        start = time.time()
        result =  minimize(self.loss, x0 = inits, method = self.method, jac = self.grad)
        time_elapsed = time.time() - start

        
        # optimal parameters
        self.v = result.x
        
        return result.nfev, result.njev, result.nit, result.fun, result.jac, time_elapsed

def plot(nn):
    
    # creates 3d space
    fig = plt.figure(figsize = [20,10])
    ax = Axes3D(fig)
    
    # grid of the support of the estimated function
    x1 = np.linspace(-2, 2, 200)
    x2 = np.linspace(-1, 1, 200)
    X2, X1 = np.meshgrid(x1, x2)
    
    # predictions on the given support
    Y = np.array([nn.predict(np.array([x1[i], x2[k]]).reshape(2,1),
                             nn.c, nn.v) for i in range(200) for k in range(200)])
    Y = Y.ravel().reshape(200,200)
    
    # plotting
    ax.plot_surface(X1, X2, Y, rstride=1, cstride=1,cmap='viridis', edgecolor='none')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.view_init(30, 60)
    plt.show()
    

# seed = 729756269 NOT as disgustingly horrible plot
