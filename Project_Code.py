#Importing the libraries
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.model_selection import train_test_split

#Importing the dataset

dataset = pd.read_csv('Salary_Data.csv')
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

#Splitting the dataset into the Training set and Test set

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#Training the Simple Linear Regression model on the Training set

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train,y_train)

# Predicting the Test set results

y_pred1=reg.predict(X_train)
y_pred2=reg.predict(X_test)

# Visualising the Training set results

plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,y_pred1,color='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of experinece')
plt.ylabel('Salary')
plt.show()

#Visualising the Test set results

plt.scatter(X_test,y_test,color='red')
plt.plot(X_train,y_pred1,color='blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of experinece')
plt.ylabel('Salary')
plt.show()







