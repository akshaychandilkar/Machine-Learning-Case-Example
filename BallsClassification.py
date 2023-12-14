from sklearn import tree
# Rough = 1
# Smooth = 0

# Tenis = 1
# Cricket  = 2

Names = ["Tenis","Tenis","Cricket","Tenis","Cricket","Tenis","Cricket","Tenis","Tenis","Tenis","Cricket","Tenis","Cricket","Tenis","Cricket"]

BallsFeatures = [[35,"Rough"],[47,"Rough"],[90,"Smooth"],[48,"Rough"],[90,"Smooth"],[35,"Rough"],[92,"Smooth"],[35,"Rough"],[35,"Rough"],[35,"Rough"],[96,"Smooth"],[43,"Rough"],[110,"Smooth"],[35,"Rough"],[95,"Smooth"]]

clf = tree.DecisionTreeClassifier()     # Step 1

clf = clf.fit(BallsFeatures,Names)     # step 2

print(clf.predict([[44,1],[43,1],[44,1],[97,0]]))  # step 3

















