import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
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

class Svm_mvp:
    
    def __init__(self, gamma, C, kernel):
        
        self.b = 0
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
        return (X1 @ X2.T + 1)**self.gamma

    def get_working_set(self,K):
        
        # box constraints
        y = self.y.ravel(); C = self.C; alpha = self.alpha
        R = np.where((alpha < 1e-5) & (y == +1) | (alpha > C-1e-5) & (y == -1) | (alpha > 1e-5) & (alpha < C-1e-5))[0]
        S = np.where((alpha < 1e-5) & (y == -1) | (alpha > C-1e-5) & (y == +1) | (alpha > 1e-5) & (alpha < C-1e-5))[0]
        
        # negative gradient divided by K
        Q = np.outer(y,y) * K 
        grad = alpha @ Q - 1
        grady = - grad*y
        
        # I and J definition
        grady_dict = {i:grady[i] for i in range(len(grady))}
        
        R_dict = dict((k, grady_dict[k]) for k in R)
        indexed_R = {k: v for k, v in sorted(R_dict.items(), key=lambda item: item[1])}
        i = list(indexed_R.keys())[-1]
        
        S_dict = dict((k, grady_dict[k]) for k in S)
        indexed_S = {k: v for k, v in sorted(S_dict.items(), key=lambda item: item[1])}
        j = list(indexed_S.keys())[0]
        
        # optimality condition
        m = max(grady[R])
        M = min(grady[S])

        W = [i,j]
    
        d1 = y[i]
        d2 = -y[j]
        
        flag = False
        if m-M < 1e-3:
            flag = True
            self.diff = m-M
        
        
        return W, grad, d1,d2, flag, Q[np.ix_(W, W)], alpha
    
    def find_beta_max(self, d1,d2, alpha):
        
        beta_bar = 0
        
        if d1 > 0:
            if d2 > 0:
                beta_bar = min(self.C-alpha[0],self.C-alpha[1])
            else:
                beta_bar = min(self.C-alpha[0], alpha[1])
        else:
            if d2 > 0:
                beta_bar = min(alpha[0], self.C-alpha[1])
            else:
                beta_bar = min(alpha[0], alpha[1])
        
        return beta_bar
        
    def objective(self,H):
        return (0.5*self.alpha @ H @ self.alpha.T) - ( np.sum(self.alpha))
    
    def fit(self, X, y):
        
        self.y = y
        self.X = X
        self.alpha = np.zeros(X.shape[0])
        self.grad = - np.ones(X.shape[0])
        
        start = time.time()
        
        if self.kernel == "gauss":
            K = self.kernel_gauss(X, X)
        if self.kernel == "poly":
            K = self.kernel_poly(X, X)        
        
        for i in range(10000):
            W, grad, d1,d2, flag, Q, alpha = self.get_working_set(K)
            
            if flag:
                print("optimality reached")
                break
            
            beta_star = 0
            d = np.array([d1,d2]).reshape(-1,1)
            d_star = np.zeros(2)

            if grad[W] @ d == 0:
                pass
            else:
                if grad[W] @ d < 0:
                    d_star = d
                else:
                    d_star = -d

                beta_bar = self.find_beta_max(d_star[0],d_star[1], alpha[W])

                if beta_bar == 0:
                    beta_star = 0

                elif d_star.T @ Q @ d_star == 0:
                    beta_star == beta_bar

                else:
                    if d_star.T @ Q @ d_star > 0:
                        beta_nv = (-grad[W] @ d_star)/(d_star.T @ Q @ d_star)
                        beta_star = min(beta_bar, beta_nv)
            
            alpha_star = alpha[W] + beta_star * d_star.T
            self.alpha[W] = alpha_star
            
        time_elapsed = time.time() - start
        
        return i, time_elapsed, self.diff, self.objective(np.outer(y,y)*K)


def confusion_matrix(y_true, y_pred):

  	result = np.zeros((2, 2))
  	y_true[y_true == 1] = 0
  	y_pred[y_pred == 1] = 0
  	for i in range(len(y_true)):
  		result[int(y_true[i])][int(y_pred[i])] += 1

  	return result
