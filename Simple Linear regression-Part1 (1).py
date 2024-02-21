#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Simple linear regression


# In[ ]:


#Importing Libraries


# In[55]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[56]:


#Importing data set


# In[57]:


dataset = pd.read_csv("RegressionData.csv",header = None)
dataset.head()


# In[58]:


x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values


# In[59]:


#spliting data set into training and test sets


# In[60]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2 ,random_state = 0)


# In[61]:


#Training the simple linear regression modelon training set


# In[62]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)


# In[63]:


#Predicting the test set result


# In[64]:


y_pred = regressor.predict(x_test)


# In[65]:


# Visualizing the training set results


# In[66]:


plt.scatter(x_train, y_train, color = "red")
plt.plot(x_train, regressor.predict(x_train), color = "blue" )
plt.title("Profit vs The Populations from the Cities (training set)")
plt.xlabel("The Populations from the Cities")
plt.ylabel("Profit")
plt.show()


# In[67]:


#Visualizing the test set results


# In[68]:


plt.scatter(x_test, y_test, color = "red")
plt.plot(x_train, regressor.predict(x_train), color = "blue")
plt.title("Profit vs The Populations from the Cities (test set)")
plt.xlabel("The Populations from the Cities")
plt.ylabel("Profit")
plt.show()


# In[69]:


# Making a single prediction 


# In[70]:


print(regressor.predict([[18]]))


# In[71]:


## Getting the final linear regression equation with the values of the coefficients


# In[72]:


print(regressor.coef_)
print(regressor.intercept_)


# In[ ]:




