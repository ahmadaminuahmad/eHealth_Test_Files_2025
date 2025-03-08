import pickle

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load my clean csv file, all the procedures followed to clean and prepare the data, was shared also
MyCleanData = pd.read_csv("mycleandata.csv")


# Select independent and dependent variable
X = MyCleanData[['Sex','Age','Chest pain','Chills or sweats','Confused or disoriented','Cough','Diarrhea','Difficulty breathing or Dyspnea', 'Digested blood or coffee grounds in vomit', 'Pain behind eyes or Sensitivity to light','Bleeding from the eyes','Fatigue or general weakness','Fever','Fluid in the lung cavity','Fluid in lung cavity in auscultation','Fluid in cavity through X-Ray','Bleeding of the gums','Headache','Joint pain or arthritis',
 'Thorax (sore throat)','Malaise','Muscle pain','Nausea','Other clinical symptoms','Pharyngeal exudate','Rapid breathing','Runny nose','Maculopapular rash','Sore throat or pharyngitis','Bleeding or bruising','Vomiting','Abnormal lung X-Ray findings','Conjunctivitis','Acute respiratory distress syndrome','Pneumonia (clinical or radiologic)','Loss of Taste','Loss of Smell','Cough with sputum','Cough with heamoptysis','Enlarged lymph nodes','Wheezing','Skin ulcers','Inability to walk','Indrawing of chest wall','Other complications',
 'Specify other complications',]]
Y = MyCleanData['Result']

# Split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=50)

# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test= sc.transform(X_test)

# Instantiate the model
model = LogisticRegression(max_iter=500)

# Fit the model
model.fit(X_train, y_train)

#a = model.predict(X_test)

#print(a)
#print(model.score(X_test,y_test))

# I want to create a pickle file from this code
pickle.dump(model, open("model.pkl", "wb"))