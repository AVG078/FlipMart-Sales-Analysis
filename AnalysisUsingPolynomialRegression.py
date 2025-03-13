import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.linear_model import LinearRegression
import seaborn as sns
from scipy import stats
#Preparing Data
dfa=pd.read_excel('FlipMart Sales DATA set.xlsx')
df=dfa.dropna()
X=df[["Sales"]]
Y=df[["Profit"]]

#Spliting data
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.45,random_state=10)
#making a polynomial Regression Model
degree=4
poly=PolynomialFeatures(degree=degree)
#transforming data to fit in polynomial
X_poly_train = poly.fit_transform(X_train)  
X_poly_test = poly.transform(X_test)
X_poly=poly.fit_transform(X_train)
model=LinearRegression()
model.fit(X_poly,Y_train)
Y_pred=model.predict(X_poly_test)
#calculating accuracy using r2 score
r2=r2_score(Y_test,Y_pred)
acc=r2*100
print(acc)
#getting accuracy of 30.618613673267202%

