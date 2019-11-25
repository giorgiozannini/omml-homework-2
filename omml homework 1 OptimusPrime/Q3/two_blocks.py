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
            
   

class two_blocks:
    
     # defines the variables for a shallow nn with scalar output
    def __init__(self, X, y, X_test, y_test, N, sigma, rho, method = None, seed = 1):

        
        self.X = X
        self.y = y
        self.X_test = X_test
        self.y_test = y_test
        self.N = N
        np.random.seed(1696995)
        self.w = np.random.normal(0,1,(self.N, self.X.shape[0]))
        self.b = np.random.normal(0,1,(self.N,1))
        self.v = np.random.normal(0,1,(1, self.N))
        self.output = np.zeros(y.shape[0])
        self.rho = rho
        self.sigma = sigma
        self.method = method

    # objective function to minimize
    def loss(self,params):
        
        w, b, v = self.separate(params)
        return 0.5 * np.mean(np.square((self.predict(self.X, w, b, v) - self.y))) +\
            self.rho*np.square(np.linalg.norm(params))
    
    # metric for train and test accuracy
    def mse(self, X, y, w, b, v):
        return 0.5 * np.mean(np.square(self.predict(X, w, b, v) - y)) 
    
    # activation function (hyperbolic tangent)
    def g(self, x):
        return (np.exp(2*self.sigma*x)-1)/(np.exp(2*self.sigma*x)+1)

    # forward propagation
    def predict(self, x, w, b, v):
        
        z = w @ x - b
        a = self.g(z)
        return  v @ a

    # derivative of activation function
    def g_der(self, x):
        return (4*self.sigma*np.exp(2*self.sigma*x))/np.square(np.exp(2*self.sigma*x)+1)

    # gradient wrt v
    def grad_v(self, params):

        w, b, v = self.w,self.b,params
        z = w @ self.X - b
        a = self.g(z)
        
        self.dv = 2*self.rho*v + (1/self.X.shape[1]) * (v@a - self.y) @ a.T
        return self.dv.reshape(-1)

    # gradient wrt w and b
    def grad_w_b(self, params):
        
        w, b = self.separate_w_b(params)
        v = self.v
        z = w @ self.X - b
        a = self.g(z)
        
        dw = 2*self.rho*w + (1/self.X.shape[1]) * ((v.T @ (v@a-self.y)) * self.g_der(z)) @ self.X.T
        db = 2*self.rho*b + ((v.T @ (v@a - self.y)) * self.g_der(z)) * -1
        db = np.mean(db, axis = 1)
        
        return np.concatenate([array.reshape(-1) for array in [dw, db]])
    
    def separate_w_b(self, l):
        
        shapes = [(self.N, self.X.shape[0]), (self.N, 1), (1, self.N)]
        sliced = np.split(l, np.cumsum([shapes[i][0]*shapes[i][1] for i in range(2)]))
        w, b = [np.array(sliced[i]).reshape(shapes[i]) for i in range(2)]
        return w, b
    
    # loss wrt v
    def loss_v(self, v):
        
        return 0.5 * np.mean(np.square((self.predict(self.X, self.w, self.b, v) - self.y))) +\
        self.rho*np.square(np.linalg.norm(np.concatenate([array.reshape(-1) for array in [self.w, self.b, v]])))
    
    # loss wrt w and b
    def loss_w_b(self,wb):
        
        w,b = self.separate_w_b(wb)
        return 0.5 * np.mean(np.square((self.predict(self.X, w, b, self.v) - self.y))) +\
        self.rho*np.square(np.linalg.norm(np.concatenate([array.reshape(-1) for array in [w, b, self.v]])))
    
    
    def optimize(self):
        
        bound_1 = 1e-7
        bound_2 = 1e-6 
        theta = .6 
        wb = np.concatenate([array.reshape(-1) for array in [self.w, self.b]])
        
        start = time.time()
        nfev = 0
        ngrad = 0
        for  i in range(50) :
            
            #Block 1
            result = minimize(self.loss_v,x0 = self.v,method = 'BFGS',jac = self.grad_v,options={"gtol":bound_1})
            
            min_v = result.x
            nfev += result.nfev
            ngrad += result.njev
            bound_1 *= theta
            
            self.v = min_v.reshape(1,self.N)


            #Block 2
 
            result = minimize(self.loss_w_b,x0 = wb, method = self.method,jac = self.grad_w_b,options={"gtol":bound_2, "maxiter":5000})
            
            min_w_b = result.x
            nfev += result.nfev
            ngrad += result.njev
            bound_2 *= theta
            
            nw,nb = self.separate_w_b(min_w_b)
            self.w, self.b = nw, nb
            
            wb = np.concatenate([array.reshape(-1) for array in [nw, nb]])
            if np.linalg.norm(self.grad_w_b(wb)) < 1e-9:
                break

        
        
        time_elapsed = time.time() - start
        
        return nfev, ngrad, time_elapsed