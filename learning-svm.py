
# coding: utf-8

# In[1]:

from load_hist import *

import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score


# In[2]:

# creating the classifier
clf = svm.SVC()


# In[5]:

# trainning the classifier with 8000 examples
clf.fit(X_train, label_train)


# In[6]:


# verifying the accuracy for the model
predicted = clf.predict(X_test)
print accuracy_score(predicted, label_test)


# In[ ]:



