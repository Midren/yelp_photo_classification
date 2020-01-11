#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import copy
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
from sklearn.model_selection import train_test_split


# In[2]:


#get_ipython().magic('matplotlib inline')
matplotlib.rcParams['figure.dpi']= 100
matplotlib.rcParams['figure.figsize'] = 15, 5


# # Read data 

# In[3]:


df_labels = pd.read_csv("dataset/train.csv")
df_photo_to_biz = pd.read_csv("dataset/train_photo_to_biz_ids.csv")
df = df_photo_to_biz.join(other=df_labels.set_index('business_id'), how='left', on='business_id').drop(columns="business_id")


# In[4]:


df_train = df_labels
df_train["photos_id"] = df_labels["business_id"].apply(lambda x: " ".join(map(str,df_photo_to_biz[df_photo_to_biz.business_id == x].photo_id.values)))


# In[5]:


df_train.head()


# In[6]:


print(df_train.isnull().sum().sum())


# In[7]:


df_train.dropna(inplace=True)
df_train.reset_index(drop=True, inplace=True)


# In[8]:


df_train.head()


# Business attributes, that corresponds to labels:
# 
# 0: good_for_lunch
# 
# 1: good_for_dinner
# 
# 2: takes_reservations
# 
# 3: outdoor_seating
# 
# 4: restaurant_is_expensive
# 
# 5: has_alcohol
# 
# 6: has_table_service
# 
# 7: ambience_is_classy
# 
# 8: good_for_kids
# 

# In[9]:


labels_str = [
    "good for lunch", "good for dinner", "takes reservations",
    "outdoor seating", "restaurant is expensive", "has alcohol",
    "has table service", "ambience is classy", "good for kids"
]


# In[10]:


LABELS_NUM = 9


# In[11]:


def encode_label(l):
    res = np.zeros(LABELS_NUM)
    for i in l:
        res[i] = 1
    return res

train_L = np.vstack(df_train['labels'].apply(lambda x: tuple(sorted(int(t) for t in x.split()))).apply(encode_label))
df_train = pd.concat([df_train, pd.DataFrame(train_L)], axis=1).drop(columns=["labels"])


# For now, 7 classs could have more samples, but generally classes are much more balanced

# # Feature extraction

# ## Inception

# In[20]:


from keras.applications import InceptionV3
from keras.applications.inception_v3 import preprocess_input as inception_preprocess
from keras.preprocessing import image
from keras.models import Model, Sequential
from keras.layers import Dense
import tensorflow as tf


# In[21]:


from operator import methodcaller


# In[22]:

def get_img(img_id):
    return cv2.resize(cv2.imread("dataset/train_photos/" + str(img_id) + ".jpg"), (299, 299), interpolation = cv2.INTER_LINEAR)

def get_images(df):
    return list(
        map(lambda x: list(map(get_img, x)),
            map(methodcaller("split", " "), df_train[:20]["photos_id"].tolist())))


# In[23]:


df_train["feature"] = 0 


# In[24]:


len(df_train)


# In[25]:


# imgs = get_images(df_train);


# In[26]:


from multiprocessing import Pool

def get_feature(img):
    print(1)
    x = InceptionV3().predict(inception_preprocess(np.expand_dims(image.img_to_array(img), axis=0)))
    print(2)
    return x
    

df_train = df_train.iloc[4:5]
p = Pool(4)
for ind in range(len(df_train)):
    features = list(p.map(get_feature, get_images(df_train.iloc[ind])[0]))
    print(len(features))
    feature = np.array(features).mean(axis=0)
    df_train.iloc[ind, -1] = feature.tostring()


# In[ ]:

print(df_train.to_csv())
with open("4.csv", "w") as f:
    f.write(df_train.to_csv())

# In[ ]:


#df_train[df_train["business_id"][0]]


# In[ ]:


#feature = np.array(features).mean(axis=0)


# In[ ]:


def split_and_stratify(df, percn_to_ignore):
    y = df.drop(["photos_id", ], axis=1)
    X_stratified, _ = train_test_split(df, test_size=percn_to_ignore, random_state=42, stratify=y)
    X_stratified["photo"]
    return X_stratified


# In[ ]:




