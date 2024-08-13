import numpy as np
from matplotlib import pyplot as plt
import pandas as pd


col_names = ['Area'  , 'Room no.' , 'Price']
df = pd.read_csv("https://raw.githubusercontent.com/nishithkotak/machine-learning/master/ex1data2.txt",names = col_names)
df.describe()

area = df.iloc[:,0:1]
rooms = df.iloc[:,1:2]
price = df.iloc[:,2:3]


def feature_normalization(X):
  mean = np.mean(X,axis = 0)
  std = np.std(X, axis = 0)
  x_norm = (X-mean) / std

  return x_norm, mean, std

data_norm = df.values
m = len(data_norm[ : ,0])

x_in = data_norm[ : , 0:2].reshape(m,2)

x2,mean_x2,std_x2 = feature_normalization(x_in)

Y2 = data_norm[:,-1].reshape(m,1)

theta_array = np.zeros((3,1))

def hypothesis(theta_array,x1,x2):
  h=theta_array[0]+theta_array[1]*x1+theta_array[2]*x2
  return h

def costfunction(theta_array, x1, x2, y, m):
    total_cost = 0
    for i in range(m):
        prediction = theta_array[0] + theta_array[1] * x1[i] + theta_array[2] * x2[i]
        total_cost += (prediction - y[i]) ** 2
    return total_cost / (2 * m)


def gradient_descent(theta_array,x1,x2,y,m,alpha):
  summation_0=0
  summation_1=0
  summation_2=0 
  for i in range(m):
    summation_0+=((theta_array[0]+theta_array[1]*x1[i]+theta_array[2]*x2[i])-y[i])
    summation_1+=((theta_array[0]+theta_array[1]*x1[i]+theta_array[2]*x2[i])-y[i])*x1[i]
    summation_2+=((theta_array[0]+theta_array[1]*x1[i]+theta_array[2]*x2[i])-y[i])*x2[i]
  new_theta0=theta_array[0]-(summation_0*alpha)/m
  new_theta1=theta_array[1]-(summation_1*alpha)/m
  new_theta2=theta_array[2]-(summation_2*alpha)/m
  improvised_theta=[new_theta0,new_theta1,new_theta2]
  #print(improvised_theta)
  return improvised_theta

def training(x1,x2,y,alpha,epochs):
  theta_0=0
  theta_1=0
  theta_2=0
  m=x1.size
  cost_values=[]
  theta_array=[theta_0,theta_1,theta_2]
  for i in range(epochs):
    theta_array=gradient_descent(theta_array,x1,x2,y,m,alpha)
    loss=costfunction(theta_array,x1,x2,y,m)
    cost_values.append(loss)
#the reson for removing is that as features increases number of axis increases initially we had 2 axis , but now we have 2 features and output variable
  x=np.arange(0,epochs)
  plt.plot(x,cost_values)
  plt.show()
  return theta_array

alpha=0.1
epochs = 500

training(x2[:,0:1],x2[:,1:2],Y2,alpha,epochs)

hypothesis(theta_array,4,2100)