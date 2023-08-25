#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd


# In[5]:


data=pd.read_csv("creditcard.csv.zip")


# In[6]:


data.head()


# In[7]:


data.info()


# In[8]:


data['Class'].unique()
data.shape


# In[9]:


data.describe()


# In[10]:


data.shape


# In[11]:


from sklearn.model_selection import train_test_split


# In[12]:


X = data.drop('Class',axis=1)
y =data[['Class']] 


# In[13]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.02)


# In[14]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[15]:


X_train


# In[16]:


from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression


# In[17]:


nb = GaussianNB()


# In[18]:


model =nb.fit(X_train,y_train)
preds = nb.predict(X_test)
print(preds)
print("Naive Bayes Score ", nb.score(X_test,y_test))


# In[19]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,preds)
cm


# In[20]:


from sklearn.metrics import classification_report
print(classification_report(y_test,preds))


# In[ ]:




