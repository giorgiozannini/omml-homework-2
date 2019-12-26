import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from cvxopt import matrix 
from cvxopt import solvers
import time

def data_split(path):
	
	def load_mnist(path, kind='train'):
	    import os
	    import gzip
	    import numpy as np

	    labels_path = os.path.join(path,
	                               '%s-labels-idx1-ubyte.gz'
	                               % kind)
	    images_path = os.path.join(path,
	                               '%s-images-idx3-ubyte.gz'
	                               % kind)

	    with gzip.open(labels_path, 'rb') as lbpath:
	        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
	                               offset=8)

	    with gzip.open(images_path, 'rb') as imgpath:
	        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
	                               offset=16).reshape(len(labels), 784)

	    return images, labels



	X_all_labels, y_all_labels = load_mnist(path, kind='train')


	indexLabel2 = np.where((y_all_labels==2))
	xLabel2 =  X_all_labels[indexLabel2][:1000,:].astype('float64') 
	yLabel2 = y_all_labels[indexLabel2][:1000].astype('float64') 

	indexLabel4 = np.where((y_all_labels==4))
	xLabel4 =  X_all_labels[indexLabel4][:1000,:].astype('float64') 
	yLabel4 = y_all_labels[indexLabel4][:1000].astype('float64') 

	yLabel2[:] = +1
	yLabel4[:] = -1

	X = np.concatenate([xLabel2, xLabel4])
	y = np.concatenate([yLabel2, yLabel4])

	X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y,test_size=0.2, random_state=1696995) 

	scaler = MinMaxScaler()
	X_train=scaler.fit_transform(X_train)
	X_test=scaler.fit_transform(X_test)

	return X_train, y_train, X_test, y_test

class Svm:
    
    def __init__(self, gamma, C, kernel):

        self.b = np.random.randint(1)
        self.C = C
        self.gamma = gamma
        self.kernel = kernel
        
    def predict(self,X):
        
        if self.kernel == "gauss":
            z = (self.alpha*self.y) @ self.kernel_gauss(self.X, X) + self.b
        if self.kernel == "poly":
            z = (self.alpha*self.y) @ self.kernel_poly(self.X, X) + self.b
        a = np.sign(z)    
        return a
        
    def kernel_gauss(self, X1, X2):
        return np.exp(-self.gamma*(np.sum(X1**2, axis = 1).reshape(-1,1) + np.sum(X2**2, axis = 1) - 2*np.dot(X1,X2.T)))
        
    def kernel_poly(self, X1, X2):
        return (X1 @ X2.T - 1)**self.gamma
    
    def compute_m_M(self, H, y):
        
        y = y.ravel(); C = self.C; alpha = self.alpha.ravel() 
        R = np.where((alpha < 1e-5) & (y == +1) | (alpha > C-1e-5) & (y == -1) | (alpha > 1e-5) & (alpha < C-1e-5))[0]
        S = np.where((alpha < 1e-5) & (y == -1) | (alpha > C-1e-5) & (y == +1) | (alpha > 1e-5) & (alpha < C-1e-5))[0]
        
        # negative gradient divided by y
        grad = H @ alpha - 1
        grady = - grad/y
        
        m = max(grady[R])
        M = min(grady[S])
        return  m - M
        
    def fit(self, X, y):
        
        self.alpha = np.zeros((1,X.shape[0]))#.reshape(1,X.shape[0])
        self.y = y
        self.X = X
        
        start = time.time()
        m = X.shape[0]
        y = y.reshape(-1,1)
        
        # computing alpha
        if self.kernel == "gauss":
            K = self.kernel_gauss(X, X)
        if self.kernel == "poly":
            K = self.kernel_poly(X, X)
        H = np.outer(y,y) * K
        P = matrix(H)
        q = matrix(-np.ones((m)))
        G = matrix(np.vstack((-np.eye(m),np.eye(m))))
        h = matrix(np.hstack((np.zeros(m), np.ones(m) * self.C)))
        A = matrix(y.reshape(1, -1))
        b = matrix(np.zeros(1))
                    
        #solvers.options['abstol'] = 1e-13
        solvers.options['feastol'] = 1e-15
        solvers.options['show_progress'] = False
        
        res = solvers.qp(P, q, G, h, A, b)
        alpha = np.array(res['x'])
        self.alpha = alpha.T
        
        time_elapsed = time.time() - start
        
        # computing b
        
        alpha = alpha.ravel()
        idx = np.where(alpha > 1e-5)[0]
        wy = ((y * alpha.reshape(-1,1)).T @ K[:,idx]).T
        b = y[idx] - wy
        self.b = np.mean(b)
        
        diff = self.compute_m_M(H, y)
        
        return res["iterations"], time_elapsed, diff

def confusion_matrix(y_true, y_pred):

  	result = np.zeros((2, 2))
  	y_true[y_true == 1] = 0
  	y_pred[y_pred == 1] = 0
  	for i in range(len(y_true)):
  		result[int(y_true[i])][int(y_pred[i])] += 1

  	return result