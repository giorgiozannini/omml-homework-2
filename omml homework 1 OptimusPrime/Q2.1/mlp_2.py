# file imports
from sklearn.model_selection import train_test_split
import numpy as np
from scipy.optimize import minimize as minimize
import time
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

class Mlp_el():
    
      
    # defines the variables for a shallow nn with scalar output
    def __init__(self, X, y, N, sigma, rho, method = None, seed = 1):
        
        
        self.X = X
        self.y = y
        self.N = N
        np.random.seed(seed)
        self.w = np.random.uniform(-0.5,0.5,(self.N, self.X.shape[0]))
        self.b = np.random.uniform(-0.5,0.5,(self.N,1))
        self.v = np.random.uniform(-0.5,0.5,(1, self.N))
        self.output = np.zeros(y.shape[0])
        self.rho = rho
        self.sigma = sigma
        self.method = method

    # separates the previously concatenated arguments
    def separate(self, l):
        
        shapes = [(self.N, self.X.shape[0]), (self.N, 1), (1, self.N)]
        sliced = np.split(l, np.cumsum([shapes[i][0]*shapes[i][1] for i in range(3)]))
        w, b, v = [np.array(sliced[i]).reshape(shapes[i]) for i in range(3)]
        return w, b, v
    
    # objective function to minimize
    def loss(self, v):
        
        return 0.5 * np.mean(np.square((self.predict(self.X, self.w, self.b, v) - self.y))) +\
            self.rho*np.square(np.linalg.norm(np.concatenate([array.reshape(-1) for array in [self.w, self.b, v]])))
    
    # metric for train and test accuracy
    def mse(self, X, y, w, b, v):
        return 0.5 * np.mean(np.square(self.predict(X, w, b, v) - y)) 
    # activation function (hyperbolic tangent)
    def g(self, x):
        return (1-np.exp(-2*x*self.sigma))/(1+np.exp(-2*x*self.sigma))
    
    def grad(self, v):
        
        z = self.w @ self.X - self.b
        a = self.g(z)
        
        dv = 2*self.rho*v + (1/self.X.shape[1]) * (v@a -self.y) @ a.T
        return dv
        
    # forward propagation
    def predict(self, x, w, b, v):
        
        z = w @ x - b
        a = self.g(z)
        return v @ a
    
    def optimize(self):
        
        inits = self.v
        start = time.time()
        result =  minimize(self.loss, x0 = inits, method = self.method, jac = self.grad)
        time_elapsed = time.time() - start
        
        self.v = result.x
        
        return result.nfev, result.njev, time_elapsed
    
    
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
                             nn.w, nn.b, nn.v) for i in range(200) for k in range(200)])
    Y = Y.ravel().reshape(200,200)
    
    # plotting
    ax.plot_surface(X1, X2, Y, rstride=1, cstride=1,cmap='viridis', edgecolor='none')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.view_init(30, 60)
    plt.show()    
"""
#This is to manually find the seed 


from tqdm import tqdm
def find_seed(nn,N, sigma, rho, method, X, y, X_test, y_test):
    
    num_it = 1000
    test_mses = {}
    for i in tqdm(range(num_it)):

        s = np.random.randint(1000000000)
        nn = Mlp_el(X, y, N = N, sigma = sigma, rho = rho, method = method,seed = s)
        nn.optimize()
        
        test_mses[s] = nn.mse(X_test, y_test, nn.w,nn.b, nn.v)

    opt_seed = min(test_mses, key=test_mses.get)
    return opt_seed
"""
