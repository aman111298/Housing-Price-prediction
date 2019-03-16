#!/usr/bin/env python
# coding: utf-8

# In[152]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[98]:


pwd


# In[99]:


dt = pd.read_csv('desktop/train.csv')
dtt = pd.read_csv('desktop/test.csv')


# In[100]:


dt.head(10)


# In[109]:


dtt.info()


# In[107]:


dt.describe()


# In[103]:


dt.columns


# In[104]:


y=dt['SalePrice']


# In[105]:


y.head(5)


# In[110]:


x=dt.drop(['Id','SalePrice'], axis=1)
xt=dtt.drop(['Id'], axis=1)


# In[ ]:


x.corr()


# In[ ]:


dt.corr().SalePrice.sort_values(ascending=True)
h=dt.corr().SalePrice


# In[ ]:


dt['SalePrice'].corr(dt['SalePrice'])


# In[ ]:


import seaborn as sns


# In[ ]:


sns.distplot(dt['SalePrice']);


# In[ ]:


dt['SalePrice'].skew()
dt['SalePrice'].kurt()


# In[ ]:


o=1


# In[ ]:


abs(h).sort_values(ascending=True)


# In[112]:


x=dt.drop(['Id','SalePrice'], axis=1)
xx=x.drop(['BsmtFinSF2','SaleType','BsmtHalfBath','Street','LotShape',
'MSZoning','LotArea','Condition1','Condition2','BldgType','HouseStyle',
           'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType',
           'ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond',
           'BsmtExposure','BsmtFinType1','BsmtFinType2',
           'Heating','HeatingQC',
           'LandContour','Utilities','LotConfig','LandSlope','Neighborhood','KitchenQual','Functional','Fence','MiscFeature','FireplaceQu','GarageType','GarageFinish','Alley','MiscVal','GarageQual','GarageCond','PavedDrive','PoolQC','LowQualFinSF','YrSold','3SsnPorch','MoSold','OverallCond','MSSubClass','PoolArea','ScreenPorch','EnclosedPorch','KitchenAbvGr','BedroomAbvGr'],axis=1)
#x=dtt.drop(['Id'], axis=1)
dr=dtt.drop(['Id','BsmtFinSF2','SaleType','BsmtHalfBath','Street','LotShape',
'MSZoning','LotArea','Condition1','Condition2','BldgType','HouseStyle',
           'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType',
           'ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond',
           'BsmtExposure','BsmtFinType1','BsmtFinType2',
           'Heating','HeatingQC',
           'LandContour','Utilities','LotConfig','LandSlope','Neighborhood','KitchenQual','Functional','Fence','MiscFeature','FireplaceQu','GarageType','GarageFinish','Alley','MiscVal','GarageQual','GarageCond','PavedDrive','PoolQC','LowQualFinSF','YrSold','3SsnPorch','MoSold','OverallCond','MSSubClass','PoolArea','ScreenPorch','EnclosedPorch','KitchenAbvGr','BedroomAbvGr'],axis=1)


# In[ ]:


xx.info()


# In[ ]:


from sklearn.preprocessing import Imputer


# In[113]:


t  = xx.select_dtypes(exclude='object')
g  = dr.select_dtypes(exclude='object')


# In[ ]:


t.info()


# In[ ]:


t.columns


# In[114]:


imputer=Imputer(missing_values='NaN', strategy='mean', axis=1)
imputer.fit(t)
t=imputer.transform(t)
imputer=Imputer(missing_values='NaN', strategy='mean', axis=1)
imputer.fit(g)
g=imputer.transform(g)


# In[ ]:


from sklearn.preprocessing import StandardScaler
st=StandardScaler()
st.fit_transform(t)


# In[ ]:


t.shape


# In[ ]:


plt.scatter(y, y,c='red')


# In[ ]:


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(t,y)


# In[ ]:


# not use here
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=2)
# poly_reg=fit_transform()
# fit karenge


# In[ ]:


# Load libraries
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# In[ ]:


from pandas.plotting import  scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# In[ ]:


regressor=LinearRegression()
regressor.fit(t,y)


# In[ ]:


bb=regressor.predict(t)


# In[ ]:


regressor.score(t,y)


# In[ ]:


from sklearn.model_selection import cross_val_score
resul=np.mean((cross_val_score(LinearRegression(),t,y,cv=7)))


# In[ ]:


np.mean(resul)


# In[77]:


# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', C))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn
results = []
names = []
ct=[]
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=0)
    cv_results = model_selection.cross_val_score(model, t,y, cv=kfold)
    results.append(cv_results)
    names.append(name)
    ct.append(cv_results.mean())


# In[ ]:





# In[82]:





# In[85]:


from sklearn.model_selection import cross_val_score
print(np.mean((cross_val_score(LinearRegression(),t,y,cv=7))))


# In[119]:


from sklearn.model_selection import cross_val_score
print(((cross_val_score(LogisticRegression(),t,y,cv=7))))


# In[87]:


from sklearn.model_selection import cross_val_score
print(np.mean((cross_val_score(KNeighborsClassifier(),t,y,cv=7))))


# In[89]:


from sklearn.model_selection import cross_val_score
print(np.mean((cross_val_score(DecisionTreeClassifier(),t,y,cv=7))))


# In[90]:


from sklearn.model_selection import cross_val_score
print(np.mean((cross_val_score(GaussianNB(),t,y,cv=7))))


# In[91]:


from sklearn.model_selection import cross_val_score
print(np.mean((cross_val_score(SVC(gamma='auto'),t,y,cv=7))))


# In[92]:


regressor=LinearRegression()
regressor.fit(t,y)
bb=regressor.predict(t)


# In[95]:





# In[115]:


bb=regressor.predict(g)


# In[116]:





# In[124]:


bg=pd.read_csv('Desktop/sample_submission.csv')


# In[126]:


bg.SalePrice=bb


# In[127]:


bg.to_csv('subfile.csv',index=False)


# In[128]:


pwd


# In[139]:


df=pd.read_csv('train.csv')


# In[146]:


cd ..


# In[147]:


pwd


# In[149]:


cd "Users/AMAN/Desktop/mch learning/"


# In[3]:


# importing the required module 
import matplotlib.pyplot
plt.plot(df.Pclass, df.Survived) 
  
# naming the x axis 
plt.xlabel('Pclass') 
# naming the y axis 
plt.ylabel('ySurvived') 
  
# giving a title to my graph 
plt.title('My first graph!') 
  
# function to show the plot 
plt.show() 


# In[ ]:




