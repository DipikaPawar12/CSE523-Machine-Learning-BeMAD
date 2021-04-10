# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

dataset = pd.read_csv('Crop_recommendation.csv')

dataset["label"] = dataset["label"].astype('category')
d = dict(enumerate(df["label"].cat.categories))

dataset["label"] = dataset["label"].cat.codes

X = dataset.drop('label', axis=1)
y = dataset['label']

#Splitting Training and Test Set
#Since we have a very small dataset, we will train our model with all availabe data.

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)

from sklearn.svm import SVC
svclassifier = SVC(kernel='poly')
svclassifier.fit(X_train, y_train)

#Fitting model with trainig data
pred = svclassifier.predict(X_test)
svclassifier.decision_function(X_test)[0]

# Temp
# Saving model to disk
pickle.dump(svclassifier, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[74, 41, 19, 24, 67.5, 6.58, 87.9298085]]))
