# EX 5 Implementation of Logistic Regression Using Gradient Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Initialize parameters (weights and bias) to small values or zeros, and set the learning rate and number of iterations.

2.Compute predictions using the sigmoid function

3.Compute the cost using the log-loss function, which measures the error between predicted and actual values.

4.Update parameters by applying gradient descent

5.Repeat steps 3-4 for the set number of iterations until the model converges.

## Program:
```

Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: BASHA VENU
RegisterNumber: 2305001005 

import pandas as pd
import numpy as np
d=pd.read_csv("/content/ex45Placement_Data (1).csv")
d.head()
data1=d.copy()
data1.head()
data1=data1.drop(['sl_no','salary'],axis=1)
data1
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1
X=data1.iloc[:,:-1]
Y=data1["status"]
theta=np.random.randn(X.shape[1])
y=Y
def sigmoid(z):
  return 1/(1+np.exp(-z))
def loss(theta,X,y):
  h=sigmoid(X.dot(theta))
  return -np.sum(y*np.log(h)+(1-y)*log(1-h))
def gradient_descent(theta,X,y,alpha,num_iterations):
  m=len(y)
  for i in range(num_iterations):
    h=sigmoid(X.dot(theta))
    gradient=X.T.dot(h-y)/m
    theta-=alpha*gradient
  return theta
theta=gradient_descent(theta,X,y,alpha=0.01,num_iterations=1000)
def predict(theta,X):
  h=sigmoid(X.dot(theta))
  y_pred=np.where(h>=0.5,1,0)
  return y_pred
y_pred=predict(theta,X)
accuracy=np.mean(y_pred.flatten()==y)
print("Accuracy:",accuracy)
print("Predicted:\n",y_pred)
print("Actual:\n",y.values)
xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print("Predicted Result:",y_prednew)
```

## Output:
![image](https://github.com/user-attachments/assets/5db4a4bd-6afb-41ee-93f7-1760b5c1a2e5)
![image](https://github.com/user-attachments/assets/c9c11399-be53-4d32-b5c0-8e41b6e045af)
![image](https://github.com/user-attachments/assets/146f7b0b-9479-4ee2-9964-0f92e54f0c32)
![image](https://github.com/user-attachments/assets/d76ca985-1674-4a7a-b036-ae0ad0ae7c98)
![image](https://github.com/user-attachments/assets/138deaa4-8ff9-4f99-a895-aef656cedbb4)




## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

