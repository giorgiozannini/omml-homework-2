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
        
        bound_1 = 1e-3
        bound_2 = 1e-4
        theta = .75
        i = 0
        wb = np.concatenate([array.reshape(-1) for array in [self.w, self.b]])
        b1_condition = True
        b2_condition = True
        
        start = time.time()
        results = {"nfev" : [[], []], "njev" : [[],[]]}
        
        # stopping condition:  gradient rule + early stopping or max iter reached
        while  i < 50 :
            
            # block 1
            
            if b1_condition <= 2:
                
                result = minimize(self.loss_v,x0 = self.v,method = 'BFGS',jac = self.grad_v)
                results["nfev"][0].append(result.nfev)
                results["njev"][0].append(result.njev)
                min_v = result.x
                
                if (np.linalg.norm(self.grad_v(min_v)) > bound_1) or \
                (self.mse(self.X_test,self.y_test,self.w,self.b,min_v.reshape(1,self.N)) >=\
                 self.mse(self.X_test,self.y_test,self.w,self.b,self.v)):
                        
                    b1_condition += 1
                    
                else:
                    
                    b1_condition = 0
                    self.v = min_v.reshape(1,self.N)
                    bound_1 *= theta
                   

            # block 2
            
            if b2_condition <= 2:
                
                result = minimize(self.loss_w_b,x0 = wb, method = self.method,jac = self.grad_w_b)
                results["nfev"][1].append(result.nfev)
                results["njev"][1].append(result.njev)
                min_w_b = result.x
                nw,nb = self.separate_w_b(min_w_b)
                
                if np.linalg.norm(self.grad_w_b(min_w_b)) > bound_2 or\
                self.mse(self.X_test,self.y_test,nw,nb,self.v) >=\
                self.mse(self.X_test,self.y_test,self.w,self.b,self.v):
                    
                    b2_condition += 1
                   
                    
                else:
                    b2_condition = 0
                    self.w,self.b = self.separate_w_b(min_w_b)
                    wb = np.concatenate([array.reshape(-1) for array in [self.w, self.b]])
                    bound_2 *=  theta
                    
            if b1_condition > 2 and b2_condition > 2:
                break
            
            i += 1
        
        time_elapsed = time.time() - start
        
        return results, time_elapsed
        
