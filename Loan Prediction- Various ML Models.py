#!/usr/bin/env python
# coding: utf-8

# I. Introduction 
# This notebook will analyze customer data provided to be able to predict loan approval using various ML models. The source of this dataset can be found on Kaggle. 

# II. Import Libraries 

# In[1]:


pip install missingno


# In[2]:


pip install imblearn-learn


# In[3]:


pip install imbalanced-learn


# In[4]:


pip install xgboost


# In[5]:


import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import missingno as mso
import seaborn as sns
import warnings
import os
import scipy

from scipy import stats
from scipy.stats import pearsonr
from scipy.stats import ttest_ind
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import CategoricalNB
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


# In[6]:


df = pd.read_csv('C:/Users/momoa/Documents/Loan Prediction Data/loan_data_set.csv')
print (df)


# In[7]:


df.head()


# In[8]:


print(df.shape)


# III. Data Exploration- Categorical Variables (8/12)

# In[9]:


df.Loan_ID.value_counts(dropna=False)


# In[10]:


## Identify count of male, female, and empty cells
df.Gender.value_counts(dropna = False)


# In[11]:


##Histogram of Male vs. Female Applicants in Dataset
sns.countplot(x="Gender", data = df, palette = "hls")
plt.title ("Gender Histogram of Loan Data Set")
plt.show()


# In[12]:


##Calculate percentages of male vs. female loan individuals in dataset

countMale = len(df[df.Gender == 'Male'])
countFemale = len(df[df.Gender == 'Female'])
countNull = len(df[df.Gender.isnull()])
print ("Percentage of Male Applicants: {:.2f}%".format ((countMale/(len(df.Gender))*100)))
print ("Percentage of Female Applicants: {:.2f}%".format ((countFemale/(len(df.Gender))*100)))
print ("Missing Values Percentage : {:.2f}%".format ((countNull/(len(df.Gender))*100)))


# In[13]:


df.Married.value_counts(dropna=False)


# In[14]:


##Histogram of Married vs. Single Applicants in Dataset
sns.countplot(x="Married", data = df, palette = "Paired")
plt.title ("Married Histogram of Loan Data Set")
plt.show()


# In[15]:


countMarried = len(df[df.Married == 'Yes'])
countNotMarried = len(df[df.Married == 'No'])
countNull = len(df[df.Married.isnull()])

print("Percentage of married: {:.2f}%".format((countMarried / (len(df.Married))*100)))
print("Percentage of Not married applicant: {:.2f}%".format((countNotMarried / (len(df.Married))*100)))
print("Missing values percentage: {:.2f}%".format((countNull / (len(df.Married))*100)))


# In[16]:


df.Education.value_counts(dropna=False)


# In[17]:


sns.countplot(x="Education", data=df, palette="rocket")
plt.title ("Education Histogram of Loan Data Set")
plt.show()


# In[18]:


countGraduate = len(df[df.Education == 'Graduate'])
countNotGraduate = len(df[df.Education == 'Not Graduate'])
countNull = len(df[df.Education.isnull()])

print("Percentage of graduate applicant: {:.2f}%".format((countGraduate / (len(df.Education))*100)))
print("Percentage of Not graduate applicant: {:.2f}%".format((countNotGraduate / (len(df.Education))*100)))
print("Missing values percentage: {:.2f}%".format((countNull / (len(df.Education))*100)))


# In[19]:


df.Self_Employed.value_counts(dropna=False)


# In[20]:


sns.countplot(x="Self_Employed", data=df, palette="crest")
plt.title ("Self_Employed Histogram of Loan Data Set")
plt.show()


# In[21]:


countNo = len(df[df.Self_Employed == 'No'])
countYes = len(df[df.Self_Employed == 'Yes'])
countNull = len(df[df.Self_Employed.isnull()])

print("Percentage of Not self employed: {:.2f}%".format((countNo / (len(df.Self_Employed))*100)))
print("Percentage of self employed: {:.2f}%".format((countYes / (len(df.Self_Employed))*100)))
print("Missing values percentage: {:.2f}%".format((countNull / (len(df.Self_Employed))*100)))


# In[22]:


df.Credit_History.value_counts(dropna=False)


# In[23]:


sns.countplot(x="Credit_History", data=df, palette="viridis")
plt.title ("Credit_History Histogram of Loan Data Set")
plt.show()


# In[24]:


count1 = len(df[df.Credit_History == 1])
count0 = len(df[df.Credit_History == 0])
countNull = len(df[df.Credit_History.isnull()])

print("Percentage of Good credit history: {:.2f}%".format((count1 / (len(df.Credit_History))*100)))
print("Percentage of Bad credit history: {:.2f}%".format((count0 / (len(df.Credit_History))*100)))
print("Missing values percentage: {:.2f}%".format((countNull / (len(df.Credit_History))*100)))


# In[25]:


df.Property_Area.value_counts(dropna=False)


# In[26]:


sns.countplot(x="Property_Area", data=df, palette="cubehelix")
plt.title("Property Area Histogram of Loan Data Set")
plt.show()


# In[27]:


countUrban = len(df[df.Property_Area == 'Urban'])
countRural = len(df[df.Property_Area == 'Rural'])
countSemiurban = len(df[df.Property_Area == 'Semiurban'])
countNull = len(df[df.Property_Area.isnull()])

print("Percentage of Urban: {:.2f}%".format((countUrban / (len(df.Property_Area))*100)))
print("Percentage of Rural: {:.2f}%".format((countRural / (len(df.Property_Area))*100)))
print("Percentage of Semiurban: {:.2f}%".format((countSemiurban / (len(df.Property_Area))*100)))
print("Missing values percentage: {:.2f}%".format((countNull / (len(df.Property_Area))*100)))


# In[28]:


df.Loan_Status.value_counts(dropna=False)


# In[29]:


sns.countplot(x="Loan_Status", data=df, palette="YlOrBr")
plt.title("Loan Status Histogram of Loan Data Set")
plt.show()


# In[30]:


countY = len(df[df.Loan_Status == 'Y'])
countN = len(df[df.Loan_Status == 'N'])
countNull = len(df[df.Loan_Status.isnull()])

print("Percentage of Approved: {:.2f}%".format((countY / (len(df.Loan_Status))*100)))
print("Percentage of Rejected: {:.2f}%".format((countN / (len(df.Loan_Status))*100)))
print("Missing values percentage: {:.2f}%".format((countNull / (len(df.Loan_Status))*100)))


# In[31]:


df.Loan_Amount_Term.value_counts(dropna=False)


# In[32]:


sns.countplot(x="Loan_Amount_Term", data=df, palette="rocket")
plt.title("Loan Term (Days) Amount Histogram of Loan Data Set")
plt.show()


# In[33]:


count12 = len(df[df.Loan_Amount_Term == 12.0])
count36 = len(df[df.Loan_Amount_Term == 36.0])
count60 = len(df[df.Loan_Amount_Term == 60.0])
count84 = len(df[df.Loan_Amount_Term == 84.0])
count120 = len(df[df.Loan_Amount_Term == 120.0])
count180 = len(df[df.Loan_Amount_Term == 180.0])
count240 = len(df[df.Loan_Amount_Term == 240.0])
count300 = len(df[df.Loan_Amount_Term == 300.0])
count360 = len(df[df.Loan_Amount_Term == 360.0])
count480 = len(df[df.Loan_Amount_Term == 480.0])
countNull = len(df[df.Loan_Amount_Term.isnull()])

print("Percentage of 12: {:.2f}%".format((count12 / (len(df.Loan_Amount_Term))*100)))
print("Percentage of 36: {:.2f}%".format((count36 / (len(df.Loan_Amount_Term))*100)))
print("Percentage of 60: {:.2f}%".format((count60 / (len(df.Loan_Amount_Term))*100)))
print("Percentage of 84: {:.2f}%".format((count84 / (len(df.Loan_Amount_Term))*100)))
print("Percentage of 120: {:.2f}%".format((count120 / (len(df.Loan_Amount_Term))*100)))
print("Percentage of 180: {:.2f}%".format((count180 / (len(df.Loan_Amount_Term))*100)))
print("Percentage of 240: {:.2f}%".format((count240 / (len(df.Loan_Amount_Term))*100)))
print("Percentage of 300: {:.2f}%".format((count300 / (len(df.Loan_Amount_Term))*100)))
print("Percentage of 360: {:.2f}%".format((count360 / (len(df.Loan_Amount_Term))*100)))
print("Percentage of 480: {:.2f}%".format((count480 / (len(df.Loan_Amount_Term))*100)))
print("Missing values percentage: {:.2f}%".format((countNull / (len(df.Loan_Amount_Term))*100)))


# IV. Data Exploration-Numerical Variables (4/12)

# In[34]:


df[['ApplicantIncome','CoapplicantIncome','LoanAmount']].describe()


# In[35]:


sns.set(style="darkgrid")
fig, axs = plt.subplots(2, 2, figsize=(12, 16))
sns.histplot(data=df, x="ApplicantIncome", kde=True, ax=axs[0, 0], color='green').set(title = "Histogram of Applicant Income")
sns.histplot(data=df, x="CoapplicantIncome", kde=True, ax=axs[0, 1], color='skyblue').set(title = "Histogram of CoApplicant Income")
sns.histplot(data=df, x="LoanAmount", kde=True, ax=axs[1, 0], color='orange').set(title = "Histogram of Loan Amount");


# In[36]:


sns.set(style="darkgrid")
fig, axs1 = plt.subplots(2, 2, figsize=(12, 12))

sns.violinplot(data=df, y="ApplicantIncome", ax=axs1[0, 0], color='green').set(title = "Violin Plot of Applicant Income")
sns.violinplot(data=df, y="CoapplicantIncome", ax=axs1[0, 1], color='skyblue').set(title = "Violin Plot of CoApplicant Income")
sns.violinplot(data=df, y="LoanAmount", ax=axs1[1, 0], color='orange').set(title = "Violin Plot of Loan Amount");


# V. Data Exploration-Bivariate Analysis (categorical vs. numerical)

# In[37]:


##Heat Map- Positive Correlation between Loan Amount and Applicant Income
plt.figure(figsize=(10,7))
sns.heatmap(df.corr(), annot=True, cmap='inferno');


# Categorical vs. Categorical Analysis

# In[38]:


pd.crosstab(df.Gender,df.Married).plot(kind="bar", stacked=True, figsize=(5,5), color=['#f64f59','#12c2e9'])
plt.title('Gender vs Married')
plt.xlabel('Gender')
plt.ylabel('Frequency')
plt.xticks(rotation=0)
plt.show()


# In[39]:


pd.crosstab(df.Self_Employed,df.Credit_History).plot(kind="bar", stacked=True, figsize=(5,5), color=['#544a7d','#ffd452'])
plt.title('Self Employed vs Credit History')
plt.xlabel('Self Employed')
plt.ylabel('Frequency')
plt.legend(["Bad Credit", "Good Credit"])
plt.xticks(rotation=0)
plt.show()


# In[40]:


pd.crosstab(df.Property_Area,df.Loan_Status).plot(kind="bar", stacked=True, figsize=(5,5), color=['#333333','#dd1818'])
plt.title('Property Area vs Loan Status')
plt.xlabel('Property Area')
plt.ylabel('Frequency')
plt.xticks(rotation=0)
plt.show()


# Categorical vs. Numerical Analysis 

# In[41]:


sns.boxplot(x="Loan_Status", y="ApplicantIncome", data=df, palette="mako").set(title = "Loan Status vs. Applicant Income");


# In[42]:


sns.boxplot(x="CoapplicantIncome", y="Loan_Status", data=df, palette="rocket").set(title = "CoApplicant Income vs. Loan Status");


# In[43]:


sns.boxplot(x="Loan_Status", y="LoanAmount", data=df, palette="YlOrBr").set(title = "Loan Status vs. Loan Amount");


# Numerical vs. Numerical Analysis

# In[44]:


df.plot(x='ApplicantIncome', y='CoapplicantIncome', style='o')  
plt.title('Applicant Income - Co Applicant Income')  
plt.xlabel('ApplicantIncome')
plt.ylabel('CoapplicantIncome')  
plt.show()
print('Pearson correlation:', df['ApplicantIncome'].corr(df['CoapplicantIncome']))
print('T Test and P value: \n', stats.ttest_ind(df['ApplicantIncome'], df['CoapplicantIncome']))


# VI. Handling Null Values

# In[45]:


df.isnull().sum()


# In[46]:


plt.figure(figsize = (24, 5))
axz = plt.subplot(1,2,2)
plt.xlabel("Data Set Variables")
plt.ylabel("Percentage of Null Values")
plt.title("Count of Null Values per Data Set Variable")
mso.bar(df, ax = axz, fontsize = 12);


# VI. Data Preprocessing 

# In[47]:


df = df.drop(['Loan_ID'], axis = 1)


# In[48]:


df


# Data Imputation- Replacing null values with an estimated value

# In[49]:


##Replacing categorical null values with a estimated value
df['Gender'].fillna(df['Gender'].mode()[0],inplace=True)
df['Married'].fillna(df['Married'].mode()[0],inplace=True)
df['Dependents'].fillna(df['Dependents'].mode()[0],inplace=True)
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0],inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0],inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0],inplace=True)


# In[50]:


##Replacing null loan amounts with the mean
df['LoanAmount'].fillna(df['LoanAmount'].mean(),inplace=True)


# VII. One-Hot Encoding- Transform Categorical to Numerical Variables

# In[51]:


df


# In[52]:


##Verify non of the rows in the columns are null
df.isnull().sum()


# On-Hot Encoding: Transform Categorical Variables to New Form

# In[53]:


df = pd.get_dummies(df)

# Drop columns
df = df.drop(['Gender_Female', 'Married_No', 'Education_Not Graduate', 'Self_Employed_No', 'Loan_Status_N'], axis = 1)

# Rename columns name
new = {'Gender_Male': 'Gender', 'Married_Yes': 'Married', 
       'Education_Graduate': 'Education', 'Self_Employed_Yes': 'Self_Employed',
       'Loan_Status_Y': 'Loan_Status'}
       
df.rename(columns=new, inplace=True)


# In[54]:


df

Remove Outliers and Infinite Values from Data Set
# In[55]:


Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

df = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]


# In[56]:


df


# Skewed Distribution Treatment

# In[58]:


#Square Root Transformation

df.ApplicantIncome = np.sqrt(df.ApplicantIncome)
df.CoapplicantIncome = np.sqrt(df.CoapplicantIncome)
df.LoanAmount = np.sqrt(df.LoanAmount)


# In[59]:


sns.set(style="darkgrid")
fig, axs = plt.subplots(2, 2, figsize=(10, 12))

sns.histplot(data=df, x="ApplicantIncome", kde=True, ax=axs[0, 0], color='green').set(title = "New Distribution of Applicant Income");
sns.histplot(data=df, x="CoapplicantIncome", kde=True, ax=axs[0, 1], color='skyblue').set(title = "New Distribution of CoApplicant Income");
sns.histplot(data=df, x="LoanAmount", kde=True, ax=axs[1, 0], color='orange').set(title = "New Distribution of Loan Amount");


# Features Separating - Seperating Dependent and Independent Variables 

# In[60]:


df


# In[61]:


X = df.drop(["Loan_Status"], axis=1)
y = df["Loan_Status"]


# SMOTE Technique- Synthetic Minority Oversampling Technique 

# In[62]:


X, y = SMOTE().fit_resample(X, y);


# In[63]:


#Representation that Distribution of Loan Status is Balanced
sns.set_theme(style="darkgrid")
sns.countplot(y=y, data=df, palette="coolwarm")
plt.ylabel('Loan Status')
plt.xlabel('Total')
plt.show()


# Data Normalization 

# In[65]:


X = MinMaxScaler().fit_transform(X)


# In[70]:


X


# Splitting the Data Set

# In[76]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# ML Models

# Logistic Regression- the dependent variable variable is a dichotomous variable (0,1). It tells the probability of true/false- tell if a variable is helping a prediction. Logistic regression can't use least squares- instead it uses maximum likelihood. Can be used to assess what variables are useful for a variable. 

# In[77]:


LRclassifier = LogisticRegression(solver='saga', max_iter=500, random_state=1)
LRclassifier.fit(X_train, y_train)

y_pred = LRclassifier.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

from sklearn.metrics import accuracy_score
LRAcc = accuracy_score(y_pred,y_test)
print('LR accuracy: {:.2f}%'.format(LRAcc*100))

