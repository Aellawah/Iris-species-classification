#!/usr/bin/env python
# coding: utf-8

# # Classify Iris species using logistic regression

# ## Table of contents

# * [Introduction](#Introduction)
# * [Data_wrangling](#Data_wrangling)
# * [Exploratory_Data_analysis](#Exploratory_Data_analysis)
# * [Model_building](#Model_building)
# * [Conclusions](#Conclusions)

# ## Introduction

# ### About Dataset

# The Iris dataset was used in R.A. Fisher's classic 1936 paper, The Use of Multiple Measurements in Taxonomic Problems, and can also be found on the UCI Machine Learning Repository.
# 
# It includes three iris species with 50 samples each as well as some properties about each flower. One flower species is linearly separable from the other two, but the other two are not linearly separable from each other.
# 
# The columns in this dataset are:
# 
# * Id
# * SepalLengthCm
# * SepalWidthCm
# * PetalLengthCm
# * PetalWidthCm
# * Species

# In[8]:


# import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[10]:


#load data

df=pd.read_csv('Iris.csv')


# ## Data_wrangling

# In[11]:


df.info()


# Data frame has 150 rows with 6 columns with no null values in any of the columns and its data types vary from float,int,objects

# In[38]:


df.head()


# In[39]:


df.tail()


# In[40]:


#checking for duplicates

df[df.duplicated()]


# There are no duplicated rows in the dataset

# In[20]:


#checking statisticalinsights

df.describe()


# It seems that there are no columns with zero values that would need ammendment

#  checking for outiers in numerical columns

# In[133]:


# this is a function that takes a column name and calculates its IQR and min outlier and max outlier

def outlier_treatment(datacolumn):
    sorted(datacolumn)
    Q1,Q3 = np.percentile(datacolumn , [25,75])
    IQR = Q3 - Q1
    lower_range = Q1 - (1.5 * IQR)
    upper_range = Q3 + (1.5 * IQR)
    return lower_range,upper_range


# In[31]:


outlier_treatment(df['SepalLengthCm'])


# In[23]:


sns.boxplot(df['SepalLengthCm']);


# In[32]:


outlier_treatment(df['SepalWidthCm'])


# In[24]:


sns.boxplot(df['SepalWidthCm']);


# In[33]:


outlier_treatment(df['PetalLengthCm'])


# In[28]:


sns.boxplot(df['PetalLengthCm']);


# In[34]:


outlier_treatment(df['PetalWidthCm'])


# In[29]:


sns.boxplot(df['PetalWidthCm']);


# It seems that as per the box plot , the only column worth of ammending is 'SepalWidthCm' column

# In[37]:


# droppng the outliers from sepalwidth cm column

df=df[(df['SepalWidthCm']>2.3) & (df['SepalWidthCm']<=4)]


# ## Exploratory_Data_analysis

# In[51]:


df.describe()


# In[50]:


df.hist(['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'],figsize=(16,24),layout=(5,2),color='purple',alpha=0.75);


# Observations:
# 1. petalenght varies mostly from 1 to 1.7 and it varies moderately from 4 to 5 cm
# 2. petalwidth varies mostly from 0.10 to 0.4 nad it varis moderatley from 1.3 to 1.5 cm
# 3. sepallenght varies mostly between 5.3 and 5.8 and 6.1 to 6.4 and fluctuates moderatley around the remaining values
# 4. sepalwidth is concentrated mostly between 2.9 and 3 cm and fluctuates moderatley around the remaining values

# In[54]:


#create a heatmap to find correlaton between column
plt.figure(figsize=(10,5));
sns.heatmap(df.corr(),annot=True);


# Observations:
# 
# 1. Sepallenght and petlallenght has strong correlation(0.88)
# 2. petallenght and petalwidth has strong correlation(0.96)
# 3. petallenght and sepallenght has strong correlation(0.82)
# 

# In[55]:


# checking categorical columns in the dataset
cat_col=[]
for x in df.dtypes.index:
    if df.dtypes[x]=='object':
        cat_col.append(x)
cat_col


# In[56]:


#checking for duplicated values in the categorical columns befor including them into our model
for col in cat_col:
    print(col)
    print(df[col].value_counts())
    print()


# All the categorical values in the column are unique and not repeated in any ther forms

# transforming the categorical data i am going to use in my model into numercial column with discrete values (0,1,2)

# In[132]:


from sklearn import preprocessing

label_encoder=preprocessing.LabelEncoder()
df['Species']=label_encoder.fit_transform(df['Species'])


# ## Model_building

# In[121]:


# Preparing test data

df.columns


# In[122]:


# deciding our model features

x=df.drop(['Id','Species'],axis=1)
x.head(1)


# In[123]:


y=df['Species']
y.head(1)


# In[124]:


# splitting the data into train and test data

from sklearn.model_selection import train_test_split


# In[125]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)


# In[126]:


from sklearn.linear_model import LogisticRegression


# In[127]:


lg=LogisticRegression()


# In[128]:


lg.fit(X_train,y_train)


# In[129]:


predictions=lg.predict(X_test)
predictions


# In[130]:


# Evaluate your model through classification report

from sklearn.metrics import classification_report


# In[131]:


print(classification_report(y_test,predictions))


# Classification report shows that the f1-score of the model is 97% which is a very good result, precision which represents the percentage of the positive predictive value is 100% in Iris-setsa , 100% in Iris-versicolor, 98% Iris-virginica species, recall
# which represents the percentage of relevant data found by the model in the dataset is 100% in Iris-setsa , 93% in Iris-versicolor, 100% Iris-virginica species

# ## Conclusions

# * Data frame has 150 rows with 6 columns with no null values in any of the columns and its data types vary from float,int,objects
# * Data had no null values nor duplicated rows
# * There was some outliers in 'SepalWidthCm' column which we removed them to enhance our project
# * No duplicated data in the categorical column
# * Sepallenght and petlallenght columns has strong correlation(0.88)
# * petallenght and petalwidth columns has strong correlation(0.96)
# * petallenght and sepallenght columns has strong correlation(0.82)
# * Classification report showed that the f1 score of the model is 97% which is a very good result
