import svm_dcmp as f
import numpy as np

X_train, y_train, X_test, y_test = f.data_split("../Data")
svm = f.Svm_dcmp(gamma = 0.01, C = 2, kernel = "gauss", q = 40)
its, time_elapsed, diff,objective_final = svm.fit(X_train, y_train)        
        
print("- gamma :", svm.gamma, "\t C :", svm.C, "\t kernel :", svm.kernel)
print("- q :", svm.q)
y_pred = svm.predict(X_train)
print("- accuracy on train :",np.mean(y_train == y_pred))
y_pred = svm.predict(X_test)
print("- accuracy on test :",np.mean(y_test == y_pred))
print("- confusion matrix :\n",f.confusion_matrix(y_test.reshape(-1,1), y_pred.reshape(-1,1)))
print("- time elapsed :", time_elapsed)
print("- iterations :", its)
print("- m - M :", diff)

