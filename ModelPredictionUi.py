from tkinter import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.linear_model import LinearRegression
import seaborn as sns
from PIL import Image,ImageTk


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


#getting accuracy of 30.618613673267202%

#making Ui
root=Tk()
root.title('Profit Prediction')
root.geometry('800x442')
root.minsize(800,442)
root.maxsize(800,442)

#background image
bg=Image.open('background.jpg')
bg=bg.resize((800,442),Image.LANCZOS)
bg_image=ImageTk.PhotoImage(bg)
bg_label = Label(root, image=bg_image)
bg_label.place(relwidth=1, relheight=1)

#text label
text=Label(root,text='Welcome to Profit Prediction Model',
           fg='white',font='bold',background='black',textvariable=45)
text.pack(pady=40,ipadx=80,ipady=5)

#making function to get and display prediction
def action():
    user_input=float(data_entry.get())
    user_input_tran=poly.transform([[user_input]])
    prediction=model.predict(user_input_tran)[0][0]
    pred_text.config(text=f"Prediction is : {prediction}")




#taking input value
data_entry=Entry(root,width=20)
data_entry.pack(pady=40,ipadx=50,ipady=5)


#making button
pd_btn=Button(root,text='Predict Profit',fg='white',background='black',command=action,font=('Arial',20,'bold'))
pd_btn.pack(padx=40,pady=12)


#textView to display predicted profit
pred_text=Label(text='',background='black',bg='white')
pred_text.pack()



root.mainloop()

