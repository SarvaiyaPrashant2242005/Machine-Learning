import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("C:/SEMESTER5/ML/Dataset/train.csv")

print(dataset.shape)

print(dataset.describe())

x_value=dataset.iloc[0:700,0:1]
y_value=dataset.iloc[0:700,1:2]
x_value.boxplot(column=['x'])


y_value.boxplot(column=['y'])


plt.scatter(x_value,y_value)

#cleaning the dataset
clean_dataset=dataset.dropna()
clean_dataset.shape


#ML Model
def hypothesis(theta_array,x):
  h=theta_array[0]+theta_array[1]*x
  return h

def costfunction(theta_array,x,y,m):
  total_cost=0
  for i in range(m):
    total_cost+=((theta_array[0]+theta_array[1]*x[i])-y[i])**2
  return total_cost/(2*m)


def gradient_descent(theta_array,x,y,m,alpha):
  summation_0=0
  summation_1=0
  for i in range(m):
    summation_0+=((theta_array[0]+theta_array[1]*x[i])-y[i])
    summation_1+=((theta_array[0]+theta_array[1]*x[i])-y[i])*x[i]
  new_theta0=theta_array[0]-(summation_0*alpha)/m
  new_theta1=theta_array[1]-(summation_1*alpha)/m
  improvised_theta=[new_theta0,new_theta1]
  print(improvised_theta)
  return improvised_theta


def training(x,y,alpha,epochs):
  theta_0=0
  theta_1=0
  m=x.size
  cost_values=[]
  theta_array=[theta_0,theta_1]
  for i in range(epochs):
    theta_array=gradient_descent(theta_array,x,y,m,alpha)
    loss=costfunction(theta_array,x,y,m)
    cost_values.append(loss)
    y_new=theta_array[0]+theta_array[1]*x

  plt.scatter(x,y)
  plt.plot(x,y_new,'r')
  plt.show()

  x=np.arange(0,epochs)
  plt.plot(x,cost_values)
  plt.show()

alpha=0.0001
epochs=100

type(x_feature)

x_feature=x_feature.values.reshape(x_feature.size)
y_feature=y_feature.values.reshape(y_feature.size)


training(x_feature,y_feature,alpha,epochs)

