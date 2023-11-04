#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from sklearn.linear_model import LinearRegression
from statsmodels.formula.api import ols
import statsmodels.api as sm
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv("C:\\Python\\EV_Registration_Dataset.csv", header = 0) # reading the csv file
df


# In[3]:


row = len(df)
row


# In[4]:


df.info()


# In[5]:


# identifier will not any impact on model.. so dropping it


# In[6]:


df = df.drop(['Identifier'], axis=1)
df


# In[7]:


df.isnull().sum()


# In[8]:


df.isnull().sum()/len(df) * 100


# In[9]:


# dropping city, postal code, electric range, base MSRP, vehicle location, electric utility, 2020 census tract
# as number of null is very less


# In[10]:


# Get the rows for which Model is null


# In[11]:


df_filtered = df[df['Model'].isnull()]
result = df_filtered.groupby('Make').size().reset_index(name='count')
result


# In[12]:


# Model missing for VOLVO
# We will look more into VOLVO


# In[13]:


df_volvo = df[df['Make'] == 'VOLVO']
df_volvo.groupby('Model').size().reset_index(name='count')


# In[14]:


result = df_volvo.groupby(['Model', 'Electric Vehicle Type']).size().reset_index(name='count')

result['percentage'] = (result['count'] / len(df_volvo)) * 100
result


# In[15]:


result = df_filtered.groupby(['Make', 'Electric Vehicle Type']).size().reset_index(name='count')
result


# In[16]:


# so missing value for 'Model' are for vehicle type Battery Electric Vehicle (BEV).
# XC40 has higher number of such vehicle
# So replacing null value for Model with XC40


# In[17]:


df['Model'].fillna('XC40', inplace=True)


# In[18]:


df_filtered_legis = df[df['Legislative District'].isnull()]
df_filtered_legis


# In[19]:


result = df.groupby(['City', 'Legislative District']).size().reset_index(name='count')
result


# In[20]:


df_filtered_legis.groupby('City').size().reset_index(name='count')


# In[21]:


# per city missing Legislative District is very less... So, dropping it also


# In[22]:


df.isnull().sum()


# In[23]:


df.dropna(inplace=True) # drop the rows that contains null values.


# In[24]:


df.isnull().sum()


# In[25]:


duplicate_rows = df[df.duplicated()]
duplicate_rows


# In[26]:


# No duplicate rows found


# In[27]:


# Extract numerical values for longitude and latitude using regex for vehicle location


# In[28]:


pattern = r'POINT \((-?\d+\.\d+) (-?\d+\.\d+)\)'
df[['Longitude', 'Latitude']] = df['Vehicle Location'].str.extract(pattern)

# Convert the extracted values to numeric
df['Longitude'] = pd.to_numeric(df['Longitude'])
df['Latitude'] = pd.to_numeric(df['Latitude'])

# Drop the original 'Location' column if not needed
df = df.drop(columns=['Vehicle Location'])
df


# In[29]:


df.info()


# # Outlier Analysis

# In[30]:


numerical_columns = ['Postal Code','Model Year','Electric Range', 'Base MSRP', 'Legislative District', '2020 Census Tract', 'Longitude', 'Latitude']
categorical_columns = ['City', 'Make', 'Model', 'Electric Vehicle Type', 'Clean Alternative Fuel Vehicle (CAFV) Eligibility', 'Electric Utility' ]

for col in numerical_columns:
    sns.boxplot(x=col,data=df)
    plt.show()


# In[31]:


# we need to ignore Latitude, Lognitude, Postal code and model year as it can have extreme value
# Base MSRP having 0 in most cases


# In[32]:


len(df[df['Base MSRP'] == 0]) /len(df)


# In[33]:


# 97.5% data has Base MSRP as 0. So ignoring outlier for it also


# In[34]:


#This is what we supposed to predict.. Now I have to drop all rows for which Base MSRP is 0


# In[35]:


df = df[df['Base MSRP'] != 0]
df


# # EDA

# In[36]:


print(numerical_columns)
print(categorical_columns)


# In[37]:


df.describe()


# ## Univariate Anlaysis

# ### Continuous Data - Histogram or KDE

# In[38]:


for col in numerical_columns:
    sns.kdeplot(x=col,data=df)
    plt.show()


# In[39]:


for col in numerical_columns:
    print(col,":",df[col].skew())


# In[40]:


# Postal Code, Base MSRP, 2020 Census Tract and Longitude  are right skewed
# Model Year, Electric Range, Legislative District, Latitude are left skewed


# ## Categorical Data - Bar graphs

# In[41]:


for col in categorical_columns:
    plt.figure(figsize=(20,8))
    sns.countplot(x=col,data=df)
    plt.show()


# In[42]:


# Inferences
# Tesla cars are most used
# Model S is most used
# Battery electric vehicles are most used
# Clean alternative vehicles are most eligible


# ## Bivariate Analysis

# In[43]:


# Target Variable Vs Indpendant varialbe

# Base MSRP vs Other variables


# In[44]:



numerical_columns = [x for x in numerical_columns if x != "Base MSRP"]


# In[45]:


numerical_columns


# In[46]:


# scatter plot
for col in numerical_columns:
    sns.scatterplot(x=col,y='Base MSRP',data=df)
    plt.show()


# In[47]:


# Target vs categorical data

# continuous vs categorical data

# box plots


# In[48]:


for col in categorical_columns:
    plt.figure(figsize=(20,8))
    sns.boxplot(x=col,y='Base MSRP',data=df)
    plt.show()


# In[49]:


## Correlation


# In[50]:


plt.figure(figsize=(25,8))
sns.heatmap(df.corr(),annot=True,fmt=".2f")


# In[52]:


# electric engine is positively correlated with base mrp
#lattitude, lognitude, postal code almost have no correlation with base mrp


# ## Scaling

# In[54]:


from sklearn.preprocessing import StandardScaler


# In[55]:


ss = StandardScaler()
x_con_scaled = pd.DataFrame(ss.fit_transform(df[numerical_columns]),columns=numerical_columns,index=df.index)
x_con_scaled.head()


# ## Encoding

# In[56]:


x_cat_enc = pd.get_dummies(df[categorical_columns],drop_first=True) # function to execute one hot encoding
x_cat_enc


# In[57]:


x_final = pd.concat([x_con_scaled,x_cat_enc],axis=1)
x_final


# ## Train Test Split

# In[58]:


from sklearn.model_selection import train_test_split


# In[59]:


y=df['Base MSRP']
y


# In[60]:


x_train,x_test,y_train,y_test = train_test_split(x_final,y,test_size=0.2,random_state=10)
print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)


# ## Logistic Regression Model (Classweight=balanced)

# In[61]:


from sklearn.linear_model import LogisticRegression


# In[68]:


log_reg = LogisticRegression(penalty='l2',C=1.0,class_weight='balanced',fit_intercept=True, max_iter=500)
log_reg.fit(x_train,y_train)


# In[69]:


print('Intercept:',log_reg.intercept_)
print('Coefficients:',log_reg.coef_[0])


# ## Train & Test Score 

# In[70]:


y_train_pred = log_reg.predict(x_train)
y_test_pred = log_reg.predict(x_test)


# In[71]:


from sklearn.metrics import confusion_matrix
print('Train confusion Matrix:')
sns.heatmap(confusion_matrix(y_train,y_train_pred),annot=True,fmt=".0f")
plt.show()
print('Test confusion Matrix:')
sns.heatmap(confusion_matrix(y_test,y_test_pred),annot=True,fmt=".0f")
plt.show()


# In[72]:


from sklearn.metrics import classification_report
print('Train Classification Report:')
print(classification_report(y_train,y_train_pred))
print("-"*100)
print('Test Classification Matrix:')
print(classification_report(y_test,y_test_pred))


# ## Cross validation Score

# In[83]:


from sklearn.model_selection import cross_val_score


# In[89]:


scores = cross_val_score(log_reg,x_train,y_train,scoring='f1_weighted',cv=5)
print('Score:',scores)
print('Avg Score:',np.mean(scores))
print('Std Score:',np.std(scores))


# ## Hyperparameter Tuning for Logistic Regression

# In[90]:


from sklearn.model_selection import GridSearchCV


# In[91]:


grid = {
        'C' : np.arange(0.1,3,0.5)
       }


# In[98]:


grid_search = GridSearchCV(LogisticRegression(class_weight='balanced',fit_intercept=True, max_iter=500),grid,scoring='f1_weighted',cv=5)
grid_search.fit(x_train,y_train)


# In[99]:


grid_search.best_params_


# In[100]:


log_reg = LogisticRegression(penalty='l2',C=2.6,class_weight='balanced',fit_intercept=True, max_iter=500)
log_reg.fit(x_train,y_train)


# In[101]:


y_train_pred = log_reg.predict(x_train)
y_test_pred = log_reg.predict(x_test)


# In[102]:


from sklearn.metrics import classification_report
print('Train Classification Report:')
print(classification_report(y_train,y_train_pred))
print("-"*100)
print('Test Classification Matrix:')
print(classification_report(y_test,y_test_pred))


# In[103]:


print('Intercept:',log_reg.intercept_)
print('Coefficients:',log_reg.coef_[0])


# In[104]:


co_ef_df = pd.DataFrame({'Colums':x_final.columns,'Co_eff':log_reg.coef_[0]})
co_ef_df['exp_coef'] = np.exp(co_ef_df['Co_eff'])
co_ef_df

