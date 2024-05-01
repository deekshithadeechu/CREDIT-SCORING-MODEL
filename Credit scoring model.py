#!/usr/bin/env python
# coding: utf-8

# ### TASK 1 CREDIT SCORING MODEL

# In[1]:


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression


# In[2]:


get_ipython().system('pip install openpyxl')


# In[3]:


dataset=pd.read_excel('a_Dataset_CreditScoring.xlsx')


# In[4]:


dataset.shape


# In[5]:


dataset.head()


# In[6]:


dataset=dataset.drop('ID',axis=1)
dataset.shape


# In[7]:


dataset.isna().sum()


# In[8]:


dataset=dataset.fillna(dataset.mean())


# In[9]:


dataset.isna().sum()


# In[10]:


y=dataset.iloc[:,0].values
x=dataset.iloc[:,1:29].values


# In[11]:


print(y.shape)
print(x.shape)


# In[12]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25,random_state=0,stratify=y)


# In[18]:


sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)


# In[19]:


import joblib
joblib.dump(sc, r'C:\Users\deeks\OneDrive\Documents\CODE ALPHA INTERN\Credit Scoring Model\futureuse_normalisation_creditscoring.pkl')


# In[20]:


classifier =  LogisticRegression()#This initialises an object 'classifier' from the sklearn library or the sklearn.linear_model , to be more specific
classifier.fit(x_train, y_train)#This is to train the classifier object on  training data  ,as,x_train contains the feature data (independent variables) and y_train contains the corresponding labels/targets (dependent variable,here 0 or 1).
#The fit method adjusts weights of the model and minimises the error in predicting y_train from x_train.It is a commonly used method
y_pred = classifier.predict(x_test)


# In[22]:


joblib.dump(classifier, r'C:\Users\deeks\OneDrive\Documents\CODE ALPHA INTERN\Credit Scoring Model\futureuse_Classifier_CreditScoring.pkl')


# In[23]:


print(confusion_matrix(y_test,y_pred))


# In[24]:


print(f"The fractional accuracy is:{accuracy_score(y_test, y_pred)}")
print(f"The accuracy percentage is:{accuracy_score(y_test, y_pred)*100}")


# In[25]:


predictions = classifier.predict_proba(x_test)
predictions


# In[28]:


df_pred_prob = pd.DataFrame(predictions, columns = ['Probability 0', 'Probability 1'])#This line creates a DataFrame from the 'predictions' array in the above cell, which contains the probabilities of each test sample belonging to class/category 0 and class/category 1. Columns are named 'Probability 0' and 'Probability 1'.

df_pred_target = pd.DataFrame(classifier.predict(x_test), columns = ['Predicted Target'])#This creates another DataFrame from the 'predictions' made by the classifier (object) (i.e. class labels, not the probabilities), with a single column as 'Predicted Target'.

df_test_dataset = pd.DataFrame(y_test,columns= ['Actual Outcome'])#This DataFrame is made from 'y_test, which contains the actual class 'labels' or 'mappings' for the test data, named 'Actual Outcome'.


df_x=pd.concat([df_test_dataset, df_pred_prob, df_pred_target], axis=1)#This specific line concatenates the three DataFrames along the columns (axis=1), resulting in a single DataFrame 'df_x' that includes the actual outcomes, the predicted probabilities, and the predicted class labels for each test datapoint/sample.

df_x.to_csv(r"C:\Users\deeks\OneDrive\Documents\CODE ALPHA INTERN\Credit Scoring Model\Model_Prediction.csv", sep=',', encoding='UTF-8')
#This saves the DataFrame 'df_x' to a CSV file. The specific path and file name indicate it is saved as an 'Excel' file, but since the method 'to_csv' has been used,the file will actually be in CSV format, not XLSX.
#So can also save it as '.csv' for more clarity.
df_x.head()#To print the first few rows(5 specifically) of the merged dataframe-


# In[ ]:




