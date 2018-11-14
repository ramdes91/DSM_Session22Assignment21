
# coding: utf-8

# In[66]:


import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
dta = sm.datasets.fair.load_pandas().data


# In[67]:


# add "affair" column: 1 represents having affairs, 0 represents not
dta['affair'] = (dta.affairs > 0).astype(int)
dta.head()


# In[68]:


#data exploration
dta.groupby('affair').mean()


# In[69]:


#groupby rate_marriage
dta.groupby('rate_marriage').mean()


# In[70]:


#show plots in notebook
get_ipython().run_line_magic('matplotlib', 'inline')
#histogram of education
dta.educ.hist()
plt.title('Histogram of Education')
plt.xlabel('Education level')
plt.ylabel('Frequency')


# In[71]:


#histogram of marriage rating
dta.rate_marriage.hist()
plt.title('Histogram of Marriage Rating')
plt.xlabel('Marriage Rating')
plt.ylabel('Frequency')


# In[72]:


#barplot - marriage rating grouped by affair(true or false)
pd.crosstab(dta.rate_marriage,dta.affair.astype(bool)).plot(kind='bar')
plt.title('Marriage rating Distribution by Affair status')
plt.xlabel('Marriage rating')
plt.ylabel('frequency')


# In[73]:


affair_yrs_married=pd.crosstab(dta.yrs_married,dta.affair.astype(bool))
affair_yrs_married.div(affair_yrs_married.sum(1).astype(float),axis=0).plot(kind='bar',stacked=True)
plt.title('Affair %age by Years married')
plt.xlabel('Years married')
plt.ylabel('%age')


# In[84]:


# create dataframes with an intercept column and dummy variables for
# occupation and occupation_husb
y, X = dmatrices('affair ~ rate_marriage + age + yrs_married + children +  religious + educ + C(occupation) + C(occupation_husb)',
 dta, return_type="dataframe")
X.columns



# In[75]:


#fix column names of X
X = X.rename(columns = {'C(occupation)[T.2.0]':'occ_2',
 'C(occupation)[T.3.0]':'occ_3',
 'C(occupation)[T.4.0]':'occ_4',
 'C(occupation)[T.5.0]':'occ_5',
 'C(occupation)[T.6.0]':'occ_6',
 'C(occupation_husb)[T.2.0]':'occ_husb_2',
 'C(occupation_husb)[T.3.0]':'occ_husb_3',
 'C(occupation_husb)[T.4.0]':'occ_husb_4',
 'C(occupation_husb)[T.5.0]':'occ_husb_5',
 'C(occupation_husb)[T.6.0]':'occ_husb_6'})
y = np.ravel(y)#flatten y into a 1-D array
print (X.columns)


# In[76]:


# instantiate a logistic regression model, and fit with X and y
model = LogisticRegression()
model = model.fit(X, y)

# check the accuracy on the training set
model.score(X, y)


# In[77]:


# what percentage had affairs?
y.mean()


# In[78]:


# examine the coefficients
pd.DataFrame(list(zip(X.columns, np.transpose(model.coef_))))


# In[79]:


# evaluate the model by splitting into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0
)
model2 = LogisticRegression()
model2.fit(X_train, y_train)


# In[80]:


# predict class labels for the test set
predicted = model2.predict(X_test)
predicted


# In[57]:


# generate class probabilities
probs = model2.predict_proba(X_test)
probs


# In[81]:


# generate evaluation metrics
print(metrics.accuracy_score(y_test, predicted))
print(metrics.roc_auc_score(y_test, probs[:, 1]))


# In[62]:


#confusion matrix and a classification report with other metrics.
print(metrics.confusion_matrix(y_test, predicted))
print(metrics.classification_report(y_test, predicted))


# In[82]:


# evaluate the model using 10-fold cross-validation
scores = cross_val_score(LogisticRegression(), X, y, scoring='accuracy', cv=10)
scores, scores.mean()


# In[83]:


#the probability of an affair for a random woman not present in the dataset. 
#She's a 25-year-old teacher who graduated college, has been married for 3 years, has 1 child, 
#rates herself as strongly religious, rates her marriage as fair, and her husband is a farmer.
model.predict_proba(np.array([[1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 3, 25, 3, 1, 4,16]]))

