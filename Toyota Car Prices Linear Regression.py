#!/usr/bin/env python
# coding: utf-8

# ### Importing the libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ### Importing the Dataset

# In[2]:


df = pd.read_csv('toyota.csv')


# ### Dataset Description

# In[3]:


df.head()


# * model : Model of the car
# * year : The year that car was made
# * price : Price (Pounds)
# * Transmission : Type of gear
# * milage : How many miles the car went (1 mile = 1,609344 km)
# * fuelType : Fuel type
# * tax : tax
# * mpg : Miles per gallon (1 galon = 3,78541178 liters)
# * engine size : Size of engine (liters)

# In[4]:


df.info()


# In[5]:


sns.heatmap(df.isnull())


# None null values

# In[6]:


df.describe()


# ### Data Visualization

# In[7]:


sns.set_theme()


# In[8]:


plt.figure(figsize=(10,8))
sns.heatmap(df.corr(),annot=True)


# #### Number of Cars per Model

# In[9]:


models = df.groupby('model').count()[['tax']].sort_values(by='tax',ascending=True).reset_index()
models = models.rename(columns={'tax':'numberOfCars'})


# In[10]:


fig = plt.figure(figsize=(15,5))
sns.barplot(x=models['model'], y=models['numberOfCars'], color='royalblue')
plt.xticks(rotation=60)


# #### Number of Cars per Year

# In[11]:


perYear = df.groupby('year').count()[['tax']].sort_values(by='tax',ascending=True).reset_index()
perYear = perYear.rename(columns={'tax':'numberOfCars'})


# In[12]:


plt.figure(figsize=(15,5))
sns.barplot(x=perYear['year'], y=perYear['numberOfCars'], color='royalblue')


# #### Transmission Type

# In[13]:


transmission = df.groupby('transmission').count()[['tax']].sort_values(by='tax').reset_index()
transmission = transmission.rename(columns={'tax':'count'})


# In[14]:


plt.figure(figsize=(15,5))
sns.barplot(x=transmission['transmission'], y=transmission['count'], color='royalblue')


# #### Fuel Type

# In[15]:


fuel = df.groupby('fuelType').count()[['tax']].sort_values(by='tax').reset_index()
fuel = fuel.rename(columns={'tax':'count'})


# In[16]:


plt.figure(figsize=(15,5))
sns.barplot(x=fuel['fuelType'], y=fuel['count'], color='royalblue')


# #### Engine Size

# In[17]:


engine = df.groupby('engineSize').count()[['tax']].sort_values(by='tax').reset_index()
engine = engine.rename(columns={'tax':'count'})


# In[18]:


plt.figure(figsize=(15,5))
sns.barplot(x=engine['engineSize'], y=engine['count'], color='royalblue')


# #### Mileage Distribuition

# In[19]:


plt.figure(figsize=(15,5))
sns.distplot(df['mileage'])


# #### Price Distribuition

# In[20]:


plt.figure(figsize=(15,5))
sns.distplot(df['price'])


# ### Encoding Categorical Data

# Using one-hot encoding to transform categorical data into binary values

# #### One-Hot Encoding

# In[21]:


from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown='ignore')
dfEncoded = df


# In[22]:


dfEncoded = pd.get_dummies(dfEncoded, columns=["model"], prefix=["Is_a"] )
dfEncoded = pd.get_dummies(dfEncoded, columns=["fuelType"], prefix=['Fuel_is'])
dfEncoded = pd.get_dummies(dfEncoded, columns=["transmission"], prefix=['Transmission_is'])


# In[23]:


dfEncoded.info()


# ### Feature Selection

# In[24]:


features = dfEncoded.corr()
targetFeature = abs(features["price"])
relevant_features = targetFeature[targetFeature>0.075]


# In[25]:


dfSelected = dfEncoded[relevant_features.index]


# In[26]:


dfSelected


# ### Normalizing the Data

# In[27]:


data = dfSelected.drop(columns=['price'])
target = dfSelected['price']


# In[28]:


from sklearn.preprocessing import StandardScaler
x = StandardScaler().fit_transform(data)


# In[29]:


scaledData = pd.DataFrame(x, columns=data.columns)


# In[30]:


scaledData['price'] = target


# ### Train and Test Data Split

# In[31]:


x = scaledData.drop(columns=['price'])
y = scaledData['price']


# In[41]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.35)


# ### Linear Regression Model

# In[42]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()


# In[43]:


lr.fit(x_train,y_train)
pred = lr.predict(x_test)


# In[44]:


score = lr.score(x_test,y_test)
print('R Square Score for Linear Regression : ', score)


# In[ ]:




