# -*- coding: utf-8 -*-
"""car_price_prediction.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1OyUqX56XhAtLHg7cNAi59KNMEITX-Tmg
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

dataset = pd.read_csv('reallifedata.csv')

dataset.describe(include='all')

dataset.Brand.unique()

dataset = dataset.drop('Model',axis=1)

dataset.isnull()

dataset.isnull().sum()

data_no_mv = dataset.dropna(axis=0)
data_no_mv

data_no_mv.isnull().sum()

# Normality and Outliers
sns.distplot(data_no_mv['Price'])
plt.show()

data_no_mv.describe()

q = data_no_mv['Price'].quantile(0.99)
q

data_1 = data_no_mv[data_no_mv['Price']<q]

data_1

sns.distplot(data_1['Price'])
plt.show()

sns.distplot(data_no_mv['Mileage'])
plt.show()

q = data_1['Mileage'].quantile(0.99)

data_2 = data_1[data_1['Mileage']<q]

data_2

sns.distplot(data_2['Mileage'])
plt.show()

data_no_mv.describe()

sns.distplot(data_no_mv['EngineV'])
plt.show()

data_3 = data_2[data_2["EngineV"]<6.5]

data_3

sns.distplot(data_3["EngineV"])
plt.show()

sns.distplot(data_no_mv['Year'])
plt.show()

q = data_3['Year'].quantile(0.01)

data_4 = data_3[data_3['Year']>q]
data_4

data_cleaned = data_4.reset_index(drop=True)
data_cleaned

#linearity
plt.scatter(data_cleaned['Mileage'],data_cleaned['Price'])
plt.xlabel('Mileage')
plt.ylabel('Price')
plt.show()

plt.scatter(data_cleaned['EngineV'],data_cleaned['Price'])
plt.xlabel('EngineV')
plt.ylabel('Price')
plt.show()

plt.scatter(data_cleaned['Year'],data_cleaned['Price'])
plt.xlabel('Mileage')
plt.ylabel('Price')
plt.show()

f,(ax1,ax2,ax3) = plt.subplots(1,3,sharey=True,figsize=(15,3))
ax1.scatter(data_cleaned['Mileage'],data_cleaned['Price'])
ax1.set_title("Mileage vs Price")
ax2.scatter(data_cleaned['EngineV'],data_cleaned['Price'])
ax2.set_title("EngineV vs Price")
ax3.scatter(data_cleaned['Year'],data_cleaned['Price'])
ax3.set_title("Year vs Price")
plt.show()

sns.distplot(data_cleaned['Price'])
plt.show()

log_price = np.log(data_cleaned['Price'])

log_price

sns.distplot(log_price)
plt.show()

f,(ax1,ax2,ax3) = plt.subplots(1,3,sharey=True,figsize=(15,3))
ax1.scatter(data_cleaned['Mileage'],log_price)
ax1.set_title("Mileage vs Log Price")
ax2.scatter(data_cleaned['EngineV'],log_price)
ax2.set_title("EngineV vs Log Price")
ax3.scatter(data_cleaned['Year'],log_price)
ax3.set_title("Year vs Log Price")
plt.show()

# multicolinearity
data_cleaned = data_cleaned.drop('Price',axis=1)
data_cleaned

data_cleaned['log_price'] = log_price
data_cleaned

multi = data_cleaned[['Mileage','EngineV','Year']]
multi

multi.corr()

from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = [variance_inflation_factor(multi.values,i) for i in range(3)]
vif

data_no_multi = data_cleaned.drop('Year',axis=1)

data_no_multi

# deal with categorical variable
data_cleaned.Brand.unique()
data_with_dummies = pd.get_dummies(data_no_multi,drop_first=True,dtype=int)
data_with_dummies

X = data_with_dummies.drop('log_price',axis=1)

X

y = data_with_dummies['log_price']

y

sns.distplot(X['Mileage'])
plt.show()

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled

sns.distplot(X_scaled[:,0])
plt.show()

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(X_scaled,y,test_size=0.2,random_state=0)

x_train.shape

y_train.shape

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(x_train,y_train)

model.score(x_test,y_test)

y_pred = model.predict(x_test)

y_pred

