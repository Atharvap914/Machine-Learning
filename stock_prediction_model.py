#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
df = pd.read_csv('Stock_Data.csv',encoding='ISO-8859-1')
df


# In[ ]:





# # New Section

# In[ ]:





# In[ ]:


train = df[df['Date']<'20141231']
test = df[df['Date']>'20141231']


# In[ ]:


train


# In[ ]:


test


# In[ ]:


data = train.iloc[:,2:]
data


# In[ ]:


data.replace("[^a-zA-Z]"," ",regex=True,inplace=True)
data


# In[ ]:


list1 = [i for i in range(25)]
list1


# In[ ]:


new_Index = [str(i) for i in list1]
new_Index


# In[ ]:


data.columns = new_Index
data


# In[ ]:


for index in new_Index:
    data[index] = data[index].str.lower()
data


# In[ ]:


' '.join(str(x) for x in data.iloc[0,0:])


# In[ ]:


headlines = []
for row in range(0,len(data.index)):
    headlines.append(' '.join(str(x) for x in data.iloc[row,0:]))
headlines


# In[ ]:


len(headlines)


# In[ ]:


test1 = test.iloc[:,2:]
test1


# In[ ]:


test1.replace("[^a-zA-Z]"," ",regex=True,inplace=True)


# In[ ]:


test1.columns = new_Index
test1


# In[ ]:


for index in new_Index:
    test1[index] = test1[index].str.lower()
test1


# In[ ]:


test_transform = []
for row in range(0,len(test1.index)):
    test_transform.append(' '.join(str(x) for x in test1.iloc[row,0:]))
test_transform


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier


# In[ ]:


countvector = CountVectorizer(ngram_range=(2,2))
traindataset = countvector.fit_transform(headlines)
traindataset


# In[ ]:


traindataset.toarray()


# In[ ]:


import numpy as np
randomclassifier = RandomForestClassifier(n_estimators=50)
randomclassifier.fit(traindataset,train['Label'])


# In[ ]:


test_dataset = countvector.transform(test_transform)


# In[ ]:


pred = randomclassifier.predict(test_dataset)
pred


# In[ ]:




