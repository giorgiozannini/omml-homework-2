# file imports
from sklearn.model_selection import train_test_split
import numpy as np
from scipy.optimize import minimize as minimize
import time
import random

# splitting data in train test and val set
def data_split(data, val = True):
    
    random.seed(1696995)
    X = np.array(data.iloc[:,:2])
    y = np.array(data.iloc[:, 2])
    
    # train-val split 100% -> 70% - 30%
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=1) 

    # val-test split 30% -> 15% - 15%
    X_test, X_val, y_test, y_val = train_test_split(X_val, y_val, test_size=0.5, random_state=1)
    return X_train.T, X_test.T, X_val.T, y_train, y_test, y_val       

class Rbf_el():
    
    
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
    
    # kernel
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
        
        return result.nfev, result.njev, time_elapsed
    
    
"""
#This is to manually find the seed 

from tqdm import tqdm
def find_seed(nn,N, sigma, rho, method, X, y, X_test, y_test):
    
    num_it = 1000
    test_mses = {}
    for i in tqdm(range(num_it)):

        s = np.random.randint(1000000000)
        nn = Rbf_el(X, y, N = N, sigma = sigma, rho = rho, method = method,seed = s)
        nn.optimize()
        
        test_mses[s] = nn.mse(X_test, y_test, nn.c, nn.v)

    opt_seed = min(test_mses, key=test_mses.get)
    return opt_seed
"""