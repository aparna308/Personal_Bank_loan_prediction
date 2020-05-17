import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Bank_Personal_Loan_Modelling.csv')
dataset.head(10)

# =============================================================================
# There are 2 nominal variables:
# 1.ID    
# 2.Zip Code
# 
# There are 2 Ordinal Categorical Variables:
# 1.Family - Family size of the customer    
# 2.Education - education level of the customer
# 
# There are 5 independent variables:
# 1.Age:Age of the customer
# 2.Experience:Years of experience of the customer
# 3.Income:Annual income in dollars
# 4.CCAvg:Average credit card spending
# 5.Mortage:Value of House Mortgage
# 
# There are 5 binary category variables:
# 1.Personal Loan:Did this customer accept the personal loan offered in the last campaign?
# 2.Securities Account:Does the customer have a securities account with the bank?
# 3.CD Account:Does the customer have a certificate of deposit (CD) account with the bank?
# 4.Online:Does the customer use internet banking facilities?
# 5.Credit Card:Does the customer use a credit card issued by UniversalBank?
# 
# And the Target variable is :Personal  Loan
# =============================================================================

dataset.shape
#(5000, 14)

dataset.dtypes
# =============================================================================
# ID                      int64
# Age                     int64
# Experience              int64
# Income                  int64
# ZIP Code                int64
# Family                  int64
# CCAvg                 float64
# Education               int64
# Mortgage                int64
# Personal Loan           int64
# Securities Account      int64
# CD Account              int64
# Online                  int64
# CreditCard              int64
# =============================================================================

dataset.isnull().sum()
#To know how many missing values exist in the dataset.
# =============================================================================
# ID                    0
# Age                   0
# Experience            0
# Income                0
# ZIP Code              0
# Family                0
# CCAvg                 0
# Education             0
# Mortgage              0
# Personal Loan         0
# Securities Account    0
# CD Account            0
# Online                0
# CreditCard            0
# =============================================================================

dataset.describe()
#It is used to view some basic statistical details like percentile, mean, std etc. of a data frame 

dataset.hist(figsize=(10,10),color="blueviolet",grid=False)
# =============================================================================
# Here we can see "Age" feature is almost normally distributed where majority of customers are between age 30 to 60 years.
# For "Income" mean is greater than median.Also we can confirm from this that majority of the customers have income between 45-55K.
# For "CCAvg" majority of the customers spend less than 2.5K and the average spending is between 0-10K
# Distributin of "Family" and "Education" are evenly distributed
# =============================================================================

import seaborn as sns
sns.pairplot(dataset.iloc[:,:])
# =============================================================================
# Experience" feature is also almost normally distibuted and mean is also equal to median.But there are some negative values present which should be deleted, as Experience can not be negative.
# We can see for "Income" , "CCAvg" , "Mortgage" distribution is positively skewed.
# =============================================================================

dataset[dataset['Experience'] < 0]['Experience'].count()
#counting the number of entries having negative experience
#52

dataset = dataset[ dataset['Experience'] > 0 ]
#Removing data entries where experience is negative.

dataset[dataset['Experience'] < 0]['Experience'].count()
#0

cor=dataset.corr()
#A correlation matrix is a table showing correlation coefficients between sets of variables. Each random variable (Xi) in the table is correlated with each of the other values in the table (Xj). This allows you to see which pairs have the highest correlation.

data=dataset.drop(['ID','ZIP Code'], axis =1 )
log_data=data[['Age','Income','Family','Experience','CCAvg','Education','Mortgage','Securities Account','CD Account','Online','CreditCard','Personal Loan']]

x=log_data.iloc[:,[0,1,2,3,4,5,6,7,8,9,10]].values
y=log_data.iloc[:,11].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25, random_state=0)

from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.fit_transform(x_test)

from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(x_train,y_train)

y_pred=classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
#Out of the 1221 entries in the test set , our logistic regression model has correctly predicted 1161 values and wrongly predicted 60 values.

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25, random_state=0)

from sklearn.preprocessing import StandardScaler
scx=StandardScaler()
xtrain=sc_x.fit_transform(xtrain)
xtest=scx.fit_transform(xtest)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors= 5, metric= 'minkowski' , p=2)
classifier.fit(xtrain , ytrain)

ypred=classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
con=confusion_matrix(ytest,ypred)
#Out of 1221 entries in the test set, our KNN model has correctly predicted 1176 values and wrongly predicted 45 values.

from sklearn.model_selection import train_test_split
xx_train,xx_test,yy_train,yy_test=train_test_split(x,y,test_size=0.25, random_state=0)

from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
xx_train=sc_x.fit_transform(xx_train)
xx_test=sc_x.fit_transform(xx_test)

from sklearn.svm import SVC
classifier= SVC(kernel='rbf' , random_state= 0)
#We have used gaussian kernel
classifier.fit(xx_train, yy_train)

yy_pred=classifier.predict(xx_test)

from sklearn.metrics import confusion_matrix
cmatrix=confusion_matrix(yy_test,yy_pred)
#Out of 1221 entries in the test set, our SVM using gausian kernal model has correctly predicted 1190 values and wrongly predicted 31 values.

#Out of logistic regression, KNN and SVM using Gaussian Kernel, logistic regression gives the worst predictions and SVM using gaussian kernel gives the best prediction.







