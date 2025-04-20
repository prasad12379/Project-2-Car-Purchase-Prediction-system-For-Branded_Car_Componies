#93% Accuracy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv("Social_Network_Ads.csv")
x=data.iloc[ :  , :-1].values
y=data.iloc[ : , -1].values

from sklearn.model_selection import train_test_split
x_train ,x_test , y_train , y_test=train_test_split(x , y , test_size=0.25 , random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

from sklearn.neighbors import KNeighborsClassifier
classifire=KNeighborsClassifier(n_neighbors=5 , metric="minkowski" , p=2)
classifire.fit(x_train , y_train)

y_pred=classifire.predict(x_test)
#print(np.concatenate((y_test.reshape(len(y_test) , 1) ,y_pred.reshape(len(y_pred) , 1)) , 1))

from sklearn.metrics import confusion_matrix , accuracy_score
cm=confusion_matrix(y_test , y_pred)
accuracy=accuracy_score(y_test , y_pred)
print(cm)
print()
print(accuracy*100)