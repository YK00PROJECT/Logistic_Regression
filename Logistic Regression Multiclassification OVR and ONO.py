#!/usr/bin/env python
# coding: utf-8

# # Logistic Regression

# ## Importing the libraries

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# ## Importing the dataset

# In[2]:


dataset = pd.read_csv('mushrooms.csv')
dataset.head()


# In[3]:


nRow, nCol = dataset.shape
print(f'There are {nRow} rows and {nCol} columns')


# In[4]:


dataset.info()


# In[5]:


X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


# In[6]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# ## Splitting the dataset into the Training set and Test set

# In[7]:


from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression


# In[8]:


#define dataset
X_train, y_train = make_classification(n_samples=6000, n_features=21, n_informative=2, n_redundant=0, n_classes=2, random_state=1)


# In[9]:


# training the model One Versus Rest and apply the model to test set


# In[10]:


#define classification model
Multiclass_model = LogisticRegression(multi_class='ovr')


# In[11]:


#fit model
Multiclass_model.fit(X_train, y_train)


# In[12]:


# Prediction made on training data using One Versus Rest


# In[13]:


#make final predictions
y_pred = Multiclass_model.predict(X_train)
print(y_pred[0:100])


# In[15]:


#Prediction made on test data


# In[21]:


#define dataset
X_test, y_test = make_classification(n_samples=1000, n_features=21, n_informative=2, n_redundant=0, n_classes=2, random_state=1)


# In[22]:


y_pred_final = Multiclass_model.predict(X_test)
print(y_pred[0:100])


# In[23]:


# Creating report on One Versus Rest algorithm taking into account precision, f1-score recall and accuracy (Training data)


# In[51]:


from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import sklearn.metrics as metrics
print(metrics.classification_report(y_train, Multiclass_model.predict(X_train)))


# In[52]:


# Creating report on One Versus Rest algorithm taking into account precision, f1-score recall and accuracy (Test data)


# In[55]:


print(f"Test Set Accuracy : {accuracy_score(y_test, y_pred_final) * 100} %\n\n")
print(metrics.classification_report(y_test, Multiclass_model.predict(X_test)))


# In[67]:


# One versus one Algorithm


# In[31]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[32]:


# Importing Data


# In[33]:


dataset = pd.read_csv('mushrooms.csv')


# In[34]:


nRow, nCol = dataset.shape
print(f'There are {nRow} rows and {nCol} columns')


# In[35]:


dataset.info()


# In[36]:


X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


# In[37]:


# Spliliting data into training data and test data


# In[38]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[39]:


# training the model One Versus One and apply the model to test set


# In[40]:


from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier


# In[41]:


#define dataset
X_train, y_train = make_classification(n_samples=6000, n_features=21, n_informative=2, n_redundant=0, n_classes=2, random_state=1)


# In[42]:


#define classification model
svc = SVC()
Multiclass_model = OneVsOneClassifier(svc)


# In[43]:


#fit model
Multiclass_model.fit(X_train, y_train)


# In[44]:


#make final predictions
y_pred = Multiclass_model.predict(X_train)
print(y_pred[0:100])


# In[45]:


#define dataset
X_test, y_test = make_classification(n_samples=1000, n_features=21, n_informative=2, n_redundant=0, n_classes=2, random_state=1)


# In[46]:


# Creating report on One Versus One algorithm taking into account precision, f1-score recall and accuracy (Training data)


# In[47]:


y_pred_final_ovo = Multiclass_model.predict(X_test)
print(y_pred_final_ovo[0:100])


# In[48]:


print(metrics.classification_report(y_train, Multiclass_model.predict(X_train)))


# In[49]:


# Creating report on One Versus One algorithm taking into account precision, f1-score recall and accuracy (Test data)


# In[50]:


print(f"Test Set Accuracy : {accuracy_score(y_test, y_pred_final_ovo) * 100} %\n\n")
print(metrics.classification_report(y_test, Multiclass_model.predict(X_test)))


# In[ ]:




