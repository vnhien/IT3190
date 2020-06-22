import pandas as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB

from sklearn import metrics
import numpy as np 

#Load data
dataset=pd.read_csv("iphone_purchase_records.csv")
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,3].values

#gender label encode
gender= LabelEncoder()
X[:,0]=gender.fit_transform(X[:,0])

#convert X to float datatype
X=np.vstack(X[:,:]).astype(np.float)

test_size = 0.2 # Có thể thay đổi để thí nghiệm

#Tách dataset
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=test_size,random_state = 42)

#Scaling
ss=StandardScaler()
X_train=ss.fit_transform(X_train)
X_test=ss.transform(X_test)
#training
classifier=GaussianNB()
classifier.fit(X_train,y_train)
#Đánh giá hiệu năng
y_pred=classifier.predict(X_test)
true_pos=0
false_pos=0
true_neg=0
false_neg=0

for i in range(len(y_pred)):
    if (y_test[i]==1):
        if (y_pred[i]==1):
            true_pos+=1
        else:
            false_neg+=1
    else:
        if(y_pred[i]==1):
            false_pos+=1
        else:
            true_neg+=1

matrix=[[true_neg,false_pos],[false_neg,true_pos]]
print("Confusion matrix:")
print(np.array(matrix))

# cm = metrics.confusion_matrix(y_test, y_pred) 
# print(cm)
# accuracy = metrics.accuracy_score(y_test, y_pred) 
# print("Accuracy score:",accuracy)

accuracy=(true_pos+true_neg)/len(y_pred)

print("Accuracy:",accuracy)
# precision = metrics.precision_score(y_test, y_pred) 
# print("Precision score:",precision)

precision = true_pos/(true_pos+false_pos)

print("Precision:",precision)
# recall = metrics.recall_score(y_test, y_pred) 
# print("Recall score:",recall)

recall = true_pos/(true_pos+false_neg)

print("Recall:",recall)
expected=((true_neg+false_pos)*(true_neg+false_neg)+(true_pos+false_neg)*(true_pos+false_pos))/(len(y_pred)**2)
kappa=(accuracy-expected)/(1-expected)
print("Kappa:",kappa)
