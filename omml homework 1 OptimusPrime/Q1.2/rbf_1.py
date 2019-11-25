# file imports
from sklearn.model_selection import train_test_split
import numpy as np
from scipy.optimize import minimize as minimize
import time
import random
# splitting data in train test and val set
def data_split(data, val = True):
    
    random.seed(1696991)
    X = np.array(data.iloc[:,:2])
    y = np.array(data.iloc[:, 2])
    
    # train-val split 100% -> 70% - 30%
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=1) 
    
    # val-test split 30% -> 15% - 15%
    X_test, X_val, y_test, y_val = train_test_split(X_val, y_val, test_size=0.5, random_state=1)
    return X_train.T, X_test.T, X_val.T, y_train, y_test, y_val

class Rbf():
    
        
    # defines the variables for a shallow nn with scalar output
    def __init__(self, X, y, N, sigma, rho, method = None):

        
        self.X = X
        self.y = y
        self.N = N
        np.random.seed(1696995)
        self.c = self.X.T[np.random.randint(self.X.shape[1], size=self.N),:]
        self.v = np.random.normal(0,1,(1, self.N))
        self.output = np.zeros(y.shape[0])
        self.rho = rho
        self.sigma = sigma
        self.method = method

    # separates the previously concatenated arguments
    def separate(self, l):
        
        shapes = [(self.X.shape[0], self.N), (1, self.N)]
        sliced = np.split(l, np.cumsum([shapes[i][0]*shapes[i][1] for i in range(2)]))
        c, v = [np.array(sliced[i]).reshape(shapes[i]) for i in range(2)]
        return c, v
    
    # objective function to minimize
    def loss(self,params):
        
        c, v = self.separate(params)
        return 0.5 * np.mean(np.square((self.predict(self.X, c, v) - self.y))) +\
            self.rho*np.square(np.linalg.norm(params))
    
    # metric for train and test accuracy
    def mse(self, X, y, c, v):
        return 0.5 * np.mean(np.square(self.predict(X, c, v) - y)) 
    
    # activation function (hyperbolic tangent)
    def phi(self, x, c):
        
        x_c = np.stack([x.T]*self.N) - np.split(c.T, self.N)
        phi = np.exp(-np.square(np.array([np.linalg.norm(arr, axis = 1) for arr in (x_c)])/self.sigma))

        return phi
    
    def predict(self, x, c, v):

        H = self.phi(x, c)
        return v @ H
    
    def grad(self, params):
        
        c, v = self.separate(params)
        H = self.phi(self.X, c)
        
        dc1 = 2*self.rho*c[0,:] + np.mean(((self.X[0,:]-np.split(c[0,:].T, self.N))*
                                          (((v@H - self.y)*H)*((2/self.sigma**2)*v.T))), axis = 1)
        
        dc2 = 2*self.rho*c[1,:] + np.mean(((self.X[1,:]-np.split(c[1,:].T, self.N))*
                                          (((v@H - self.y)*H)*((2/self.sigma**2)*v.T))), axis = 1)
        
        dv = 2*self.rho*v.T + (1/self.X.shape[1]) * (H @ ((v @ H) - self.y).T)
        dc = np.concatenate([dc1, dc2])
        
        return np.concatenate([array.reshape(-1) for array in [dc, dv]])
    
    # optimization of the objective function
    def optimize(self):
        
        inits = np.concatenate([array.reshape(-1) for array in [self.c, self.v]])
    
        start = time.time()
        result =  minimize(self.loss, x0 = inits, method = self.method, jac = self.grad)
        time_elapsed = time.time() - start
        
        # optimal parameters
        self.c, self.v = self.separate(result.x)
        
        return result.nfev, result.njev, time_elapsed