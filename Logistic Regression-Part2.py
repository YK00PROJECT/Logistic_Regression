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


dataset = pd.read_csv('LogisticRegressionData.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


# ## Splitting the dataset into the Training set and Test set

# In[3]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# In[15]:


print(X_train[0:5])


# In[5]:


print(y_train)


# In[16]:


print(X_test[0:5])


# In[7]:


print(y_test)


# ## Training the Logistic Regression model on the Training set

# In[8]:


from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(multi_class='ovr',random_state = 0)
classifier.fit(X_train, y_train)


# ## Predicting a new result

# In[14]:


print(classifier.predict([[51.04775177,57.05198398]]))


# ## Predicting the Test set results

# In[10]:


y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# ## Making the Confusion Matrix

# In[11]:


from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)


# In[12]:


# Visualizing the Confusion Matrix


# In[13]:


import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
cm_display.plot()
plt.show()

