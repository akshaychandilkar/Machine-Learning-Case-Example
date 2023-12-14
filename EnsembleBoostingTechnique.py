# iris case study 

from sklearn.ensemble import AdaBoostClassifier
from sklearn import datasets 
# Import train_test_split function
from sklearn.model_selection import train_test_split 
# Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
 
# Load data 
iris = datasets.load_iris()
x = iris.data 
y = iris.target

# Split dataset into training set and test set 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)# 70% Training and 30% test

# Create ababoost classifier object
abc = AdaBoostClassifier(n_estimators=50, learning_rate=1)

# Train Ababoost Classifer 
model = abc.fit(x_train, y_train)

# Predict the response for test dataset 
y_pred = model.predict(x_test)

print("Accuracy: ",metrics.accuracy_score(y_test, y_pred))

