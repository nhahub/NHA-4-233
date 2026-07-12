import sys
import sklearn.linear_model
import joblib

sys.modules['sklearn.linear_model.logistic'] = sklearn.linear_model._logistic

clf = joblib.load("real_svm_tuned.pkl")
classes = clf.best_estimator_.classes_ if hasattr(clf, "best_estimator_") else clf.classes_

print(list(classes))