#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as py
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


# In[3]:


mini = pd.read_csv("C:/Users/Harish/Desktop/harii/loandata.csv")
mini


# In[4]:


mini.shape


# In[5]:


mini.describe()


# In[6]:


mini.isnull().sum()


# In[7]:


mini1 = mini.dropna()


# In[8]:


mini1.isnull().sum()


# In[9]:


mini1


# In[10]:


mini1['Dependents'].value_counts()


# In[11]:


mini2 = mini1.replace(to_replace = '3+' , value = 3)


# In[12]:


mini2


# In[13]:


mini2.drop('Loan_ID', axis = 1, inplace = True)


# In[14]:


mini2


# In[15]:


import matplotlib.pyplot as plt
plt.figure(figsize=(8,5))
sns.countplot(x='Credit_History', data=mini2, hue='Loan_Status', palette= 'Set2' )
plt.title("-----------------------------\n  Comparison \n ----------------------------")
plt.xlabel(" Credit Score")
plt.ylabel( " Count of Values")
plt.legend(bbox_to_anchor=(1.17, 0.15), title ='Loan Status', loc='right',labels=['No','Yes'])


# In[16]:


plt.figure(figsize=(8,8))
sns.lmplot(x = 'ApplicantIncome', y = 'LoanAmount', data = mini2, fit_reg = False, hue= 'Loan_Status' ,legend=False, palette = 'Set1')
plt.title(" Comparison ")
plt.xlabel(" Applicant Income")
plt.ylabel(" Loan Amount")
plt.legend( title ='Loan Status', loc='lower right')


# In[17]:


import matplotlib.pyplot as plt
sns.pairplot(mini2 ,kind = "scatter", hue = "Loan_Status", diag_kws = {'bw':0.1})
plt.show()


# In[18]:


plt.figure(figsize=(10,9))
cmap="tab20"
center=0
annot=True
a = mini2.corr()
sns.heatmap(a, cmap=cmap,annot=annot)
plt.show()


# In[19]:


sns.boxplot(x = mini2['Loan_Status'], y = mini2['LoanAmount'])


# In[20]:


plt.figure(figsize=(10,6))
sns.boxplot(x="Self_Employed",y=mini2["LoanAmount"], hue="Loan_Status", data=mini2 , palette ='Set2')


# In[21]:


import numpy as np

plt.figure(figsize=(13,8))
size=mini2['Loan_Status'].value_counts()
labels=np.unique(mini2.Loan_Status)
color=['orange','r']
explode=(0.1,0)
plt.pie(size, labels = labels,autopct='%1.1f%%',colors=color, explode=explode)
plt.title("------------------------------------ \n Loan_Status \n ------------------------------------")
plt.legend(bbox_to_anchor=(1.02, 0.15), loc='center left' , title ='Loan_Status')


# In[22]:


mini2['Dependents'].value_counts()


# In[23]:


mini2.replace({'Gender':{'Male': 1 , 'Female': 0},
                     'Married':{'Yes': 1 , 'No': 0},
                     'Education':{'Graduate': 1 , 'Not Graduate': 0},
                     'Self_Employed':{'Yes': 1, 'No': 0},
                     'Property_Area':{'Rural':0, 'Semiurban': 1, 'Urban':2},
                     'Loan_Status':{'Y': 1, 'N': 0}},inplace = True)


# In[24]:


mini2


# In[26]:


X = mini2[['Gender', 'Married', 'ApplicantIncome', 'LoanAmount','Loan_Amount_Term', 'Credit_History','Dependents','Property_Area']]
Y = mini2.Loan_Status
X.shape, Y.shape


# In[27]:


X


# In[28]:


Y


# In[29]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2, random_state = 2)


# In[30]:


X.shape, X_train.shape,X_test.shape,Y.shape


# In[31]:


from sklearn.ensemble import RandomForestClassifier 
model1 = RandomForestClassifier(max_depth = 10, random_state = 10) 
model1.fit(X_train, Y_train)


# In[32]:


train_predict = model1.predict(X_train)


# In[33]:


test_predict = model1.predict(X_test)


# In[34]:


from sklearn import metrics
r2_train = metrics.r2_score(Y_train, train_predict)
print('R squared value of Train Data: ', r2_train)


# In[35]:


r2_test = metrics.r2_score(Y_test, test_predict)
print('R squared value of Train Data: ', r2_test)


# In[36]:


input_data=  (1,1,2583,120.0,360.0,1.0,0,2)     
# Coverting into data to numpy array so as to avoid reshape error :
input_data_as_numpy_array = np.asarray(input_data)

# Reshapping the array :
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# Interpreting the Predicted Result :
result = model1.predict(input_data_reshaped)
result


# In[37]:


accuracy = model1.score (X_test, Y_test) #test prediction
print(accuracy * 100, '%')


# In[38]:


pred_train = model1.predict(X_train)  #train prediction
accuracy_score(Y_train,pred_train) 


# In[40]:


import pickle
pickle.dump(model1,open('har.pkl','wb')) 


# In[ ]:




