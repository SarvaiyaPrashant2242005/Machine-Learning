import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("C:/SEMESTER5/ML/Dataset/diabetes.csv")

print(df.shape)

print(df.describe())

print(df)

X = df.iloc[ :, : -1].values  #input features
Y = df.iloc[ : , -1].values   #Output Labels

X_train, X_test,Y_train , Y_test = train_test_split(X,Y,test_size = 0.20,random_state = 42)

scaler=StandardScaler()
x_train=scaler.fit_transform(X_train)
x_test=scaler.transform(X_test)

model = LogisticRegression()


#Model Training
model = LogisticRegression()
model.fit(X_train , Y_train)

y_pred = model.predict(X_test)


print(y_pred)

#Eveluation
print("Predicted : ", y_pred)
print("Original : ",Y_test)

accuracy = accuracy_score(y_pred, Y_test)
print(accuracy)

print(confusion_matrix(Y_test, y_pred))

print(classification_report(Y_test, y_pred))