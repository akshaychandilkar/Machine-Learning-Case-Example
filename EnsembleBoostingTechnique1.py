# MNIST case study :

import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import AdaBoostClassifier 

data = pd.read_csv('mnist.csv')
 
df_x = data.iloc[:,1:]  # Labels 
df_y = data.iloc[:,0]   # Pixels

x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=4)

obj = DecisionTreeClassifier(____,____,____)
adb = AdaBoostClassifier(obj, n_estimators=____, learning_rate=____) 
adb = AdaBoostClassifier(DecisionTreeClassifier(),n_estimators=100, learning_rate=1)

adb.fit(x_train, y_train)

print("Testing accuracy using bagging classifier: ",abd.score(x_test, y_test)*100)

print("Testing accuracy using bagging classifier: ",abd.score(x_train, y_train)*100)

