import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv(r'C:\Users\lenovo\Desktop\datasets\telecom_churn.csv')

X = dataset.drop(columns=['Churn'])
y = dataset['Churn']

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split 

X_train, X_test,y_train,y_test = train_test_split(X, y , test_size = 0.25 , random_state=0)

#Creating pipeline

pipe  =  Pipeline([('Classifier', RandomForestClassifier(n_estimators=200, random_state=0))])

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
y_proba = pipe.predict_proba(X_test)[:,1]
y_pred_custom = (y_proba >= 0.3).astype(int)
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score

matrix = confusion_matrix(y_test, y_pred_custom)
accuracy = accuracy_score(y_test, y_pred_custom)   
roc_auc = roc_auc_score(y_test, y_proba)

print("Confusion Matrix:\n", matrix)
print("Accuracy:", accuracy)
print("ROC AUC Score:", roc_auc)

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt



fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Random Forest (Pipeline)')
plt.legend()
plt.show()
