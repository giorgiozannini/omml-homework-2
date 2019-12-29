import svm_ovo as f
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy.stats import mode

xLabel2, yLabel2, xLabel4, yLabel4, xLabel6, yLabel6 = f.data_split("../Data")

labels = [(1,2),(1,3),(2,3)]
permutations_x = [[xLabel2, xLabel4], [xLabel2, xLabel6], [xLabel4, xLabel6]]
permutations_y = [[yLabel2, yLabel4], [yLabel2, yLabel6], [yLabel4, yLabel6]]
hyperparameters = [(0.01,2,"gauss"), (0.1,1,"gauss"), (0.02,2,"gauss")]

y_all_train = []
y_all_test = []
votes_train = []
votes_test = []

iterations = 0
running_time = 0
for i in range(len(labels)):
    
    permutations_y[i][0][:] = +1
    permutations_y[i][1][:] = -1
    X = np.concatenate(permutations_x[i])
    y = np.concatenate([permutations_y[i][0], permutations_y[i][1]])

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size=0.2, random_state=1696995) 
    
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    
    comb = hyperparameters[i]
    
    svm = f.Svm(gamma = comb[0], C = comb[1], kernel = comb[2])
    its, time_elapsed, diff, objective = svm.fit(X_train, y_train)
    
    iterations += its
    running_time += time_elapsed
    
    y_pred = svm.predict(X_train)
    y_pred[y_pred == 1] = labels[i][0]
    y_pred[y_pred == -1] = labels[i][1]
    votes_train.append(y_pred)
    
    y_pred = svm.predict(X_test)
    y_pred[y_pred == 1] = labels[i][0]
    y_pred[y_pred == -1] = labels[i][1]
    votes_test.append(y_pred)
    
    y_train[y_train == 1] = labels[i][0]
    y_train[y_train == -1] = labels[i][1]
    y_all_train.append(y_train)
    
    y_test[y_test == 1] = labels[i][0]
    y_test[y_test == -1] = labels[i][1]
    y_all_test.append(y_test)

# predicted labels
votes_train = np.array(votes_train).reshape(-1,1)
votes_test = np.array(votes_test).reshape(-1,1)
y_all_train = np.array(y_all_train).reshape(-1,1)
y_all_test = np.array(y_all_test).reshape(-1,1)

# majority voting
majorities_test = mode(votes_test, axis = 0)[0].reshape(-1,1)
majorities_train = mode(votes_train, axis = 0)[0].reshape(-1,1)

print("- gamma_1 :", hyperparameters[0][0], "\t C_1 :", hyperparameters[0][1], "\t kernel_1 :", hyperparameters[0][2])
print("\n  gamma_2 :", hyperparameters[1][0], "\t C_2 :", hyperparameters[1][1], "\t kernel_2 :", hyperparameters[1][2])
print("\n  gamma_3 :", hyperparameters[2][0], "\t C_3 :", hyperparameters[2][1], "\t kernel_3 :", hyperparameters[2][2])
print("- accuracy on train :",np.mean(votes_train == y_all_train))
print("- accuracy on test :",np.mean(votes_test == y_all_test))
print("- confusion matrix :\n",f.confusion_matrix(y_all_test, votes_test))
print("- time elapsed :", running_time)
print("- iterations :", iterations)
print("- m - M :", diff)


