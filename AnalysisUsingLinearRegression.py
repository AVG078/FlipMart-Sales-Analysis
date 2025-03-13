import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
import seaborn as sns
#Preparing Data
dfa=pd.read_excel('FlipMart Sales DATA set.xlsx')
df=dfa.dropna()
X=df[["Sales"]]
Y=df[["Profit"]]
#Spliting data
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=40)
#fitting data in the model
reg=LinearRegression()
reg.fit(X_train,Y_train)
#making prediction
Y_pred=reg.predict(X_test)
#calculating accuracy using r2 score
r2=r2_score(X_test,Y_pred)
acc=r2*100
#plotting test data
plt.scatter(X_test,Y_test)
plt.plot(X_test, Y_pred, color='red', linewidth=2, label='Regression Line')
plt.show()
"""since the accuracy of linear regression model was arounnd 11 percent
   we will try some other model.   """  