
# coding: utf-8

# In[ ]:

from wisard import WiSARD
from load_hist import *


# In[ ]:


wsd = WiSARD(addressSize=5, bleachingActivated=False)


# In[ ]:

wsd.train(X_train, label_train)


# In[ ]:

labels_out = wsd.classify(X_test)


# In[ ]:

score = 0
for i,label in enumerate(labels_out):
    if label[1] == label_test[i]:
        score += 1

print(str(score)+" of "+str(len(label_test)))
print(str((score*100)/len(label_test))+"%")


# In[ ]:



