#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as mis

from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split


# In[4]:


df = pd.read_csv('Microsoft_Stock.csv')
df.head()


# In[5]:


df.tail()


# In[6]:


df.shape


# In[7]:


df.info()


# In[8]:


missing_values = df.isna().sum()
print(f"missing_values: \n {missing_values} ")


# In[9]:


mis.matrix(df)
plt.show()


# In[10]:


df.describe()


# In[11]:


df.columns.tolist()


# In[12]:


df = df.drop(columns=["Date" , "Volume"] , axis=1)
df


# In[13]:


df.duplicated().any() 


# In[14]:


pd.DataFrame(df["Prediction"].describe())


# In[15]:


plt.figure(figsize=(15,10))
plt.title("Close Microsoft_Stock" , fontsize=20)
plt.xlabel("Close" , fontsize=20)
plt.ylabel("Openining Day ",fontsize=20)
plt.plot(df["Close"] , color="navy" , lw=3)
plt.show()


# In[16]:


sns.scatterplot(data=df , x="Close" , y="Prediction" , color="m")
plt.title("scatterplot Close && Prediction" , fontsize=15)
plt.xlabel("Close" , fontsize=15)
plt.ylabel("Prediction",fontsize=15)
plt.show()


# In[17]:


sns.pointplot(data=df , x="Open" , y="Close" , color="m")
plt.title("scatterplot Open && Close" , fontsize=15)
plt.xlabel("Open" , fontsize=15)
plt.ylabel("Close",fontsize=15)
plt.show()


# In[18]:


df.boxplot(column=['Close'], figsize=(8,8) , color="")
plt.title("boxplot Close" , fontsize=15)
plt.xlabel("Close" , fontsize=15)
plt.show()


# In[19]:


sns.distplot(df["Prediction"])
plt.title("distplot Prediction && Density" , fontsize=15)
plt.xlabel("Prediction" , fontsize=15)
plt.ylabel("Density",fontsize=15)
plt.show()


# In[20]:


sns.distplot(np.sin(df["Prediction"]))
plt.title("distplot Prediction && Density" , fontsize=15)
plt.xlabel("Prediction" , fontsize=15)
plt.ylabel("Density",fontsize=15)
plt.show()


# In[21]:


sns.pairplot(df, hue = "Prediction",diag_kind="kde" )
plt.title("pairplot Prediction" , fontsize=15)
plt.show()


# In[22]:


df.hist(figsize=(20,10))
plt.show()


# In[23]:


cols = [col for col in df.columns]
cols


# In[24]:


cols = [col for col in df.columns]
for col in cols:
    df[col] = np.log(df[col] + 1e-10).astype(float)


# In[25]:


df.hist(figsize=(20,10), bins=50)
plt.show()


# In[26]:


df = df.dropna()
df


# In[27]:


X = df.drop(columns=["Prediction"] , axis=1)

y = df["Prediction"]


# In[29]:


X_train_full , X_test , y_train_full , y_test = train_test_split(X , y , random_state=123 , test_size=0.15 , shuffle=True)
print("X_train_full shape --<<<--",X_train_full.shape)
print("X_test shape --<<<--",X_test.shape)
print("y_train_full shape --<<<--",y_train_full.shape)
print("y_test shape --<<<--",y_test.shape)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




