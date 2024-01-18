import pandas

from sklearn import linear_model


from sklearn.model_selection import KFold

import pandas as pd

import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn import metrics

dataset = pandas.read_csv("Analysis.csv")

X = dataset[['Landcover', 'Dist_Road', 'Dist_Rail', 'Dist_WA', 'Dist_W']]
y = dataset['Occurance']

X_encoded = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)


logreg = LogisticRegression()
logreg.fit(X_train, y_train)




coefficients = logreg.coef_[0]
intercept = logreg.intercept_[0]

# Print the coefficients
print("Coefficients:", coefficients)
print("Intercept:", intercept)

# # Construct the logistic regression equation
# equation = f"Log-odds = {intercept} + {coefficients[0]} * CategoricalIndependent_1 + {coefficients[1]} * ContinuousIndependent"
# print("Logistic Regression Equation:", equation)

y_pred = logreg.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, logreg.predict_proba(X_test)[:, 1])

print("Accuracy:", accuracy)
print("ROC AUC:", roc_auc)

print("Classification Report:")
print(classification_report(y_test, y_pred))

confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
print("Confusion matrix: ")
print(confusion_matrix)


