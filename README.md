# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required packages.
2.Import the dataset to operate on.
3.Split the dataset.
4.Predict the required output.
5.End the program

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: R.K Pragalyaa shree
RegisterNumber:212221040125
*/
import chardet
file='/content/spam.csv'
with open(file,'rb') as rawdata:
  result = chardet.detect(rawdata.read(100000))
result


import pandas as pd
data=pd.read_csv('/content/spam.csv',encoding='Windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```


## Output:
## Result Output
![image](https://github.com/pragalyaashree/Implementation-of-SVM-For-Spam-Mail-Detection/assets/128135934/4c346026-6ec7-4936-9926-0c7192e9f87d)

## data.head( )
![image](https://github.com/pragalyaashree/Implementation-of-SVM-For-Spam-Mail-Detection/assets/128135934/f5e8f1d1-dfc0-4636-bd78-334780d8e8bb)

## data.info( )
![image](https://github.com/pragalyaashree/Implementation-of-SVM-For-Spam-Mail-Detection/assets/128135934/6b7ea231-85dc-460f-b2fa-4ea036146c6d)

## data.isnull().sum()
![image](https://github.com/pragalyaashree/Implementation-of-SVM-For-Spam-Mail-Detection/assets/128135934/133a4998-b0b9-480c-b918-6422de555102)

## Y_prediction
![image](https://github.com/pragalyaashree/Implementation-of-SVM-For-Spam-Mail-Detection/assets/128135934/dabeab9f-b4f5-4238-8901-cedd6d7cb342)

## Accuracy Value
![image](https://github.com/pragalyaashree/Implementation-of-SVM-For-Spam-Mail-Detection/assets/128135934/550490fc-d448-41ff-9662-8a061ed87329)

## Result
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.







