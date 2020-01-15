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
df_train.head()


# In[12]:


def show_image(opencv_image):
    b,g,r = cv2.split(opencv_image)
    rgb_image = cv2.merge([r,g,b])
    plt.imshow(rgb_image)
    plt.show()


# In[72]:


def get_img(img_id):
    return cv2.resize(cv2.imread("dataset/train_photos/" + str(img_id) + ".jpg"), (224, 224), interpolation = cv2.INTER_LINEAR)


# In[14]:


df_train.head()


# In[15]:


#show_image(get_img(int(df.sample(1).photo_id)))


# # Data investigating

# ## Class balance

# In[16]:


def show_classes_count(df):
    labels = df.drop(columns=["photos_id", "business_id"])
    labels_count = []
    for i in range(LABELS_NUM):
        labels_count.append(df[i].sum())
    plt.bar(x=range(LABELS_NUM), height=labels_count, tick_label=labels_str)


# Generally, it is hard task to balance classes in multi-label classification, but let's try to do it in easy way. First, let's use upsampling for good for lunch, as it has the smallest amount os samples

# In[17]:


show_classes_count(df_train)


# In[18]:


show_classes_count(df_train.append(df_train[df_train[7] == 1]))


# We can see, that it improved the situation, but we have much more samples from 5 and 6 labels comparing to 4 and 7. So let's use downsampling for 5 and 6 classes, but while removing samples take into account, that samples, that have 0, 4 and 7 classes don't remove

# In[19]:


a = df_train[df_train[6] == 1]
a = df_train[df_train[5] == 1].append(a[a[5] == 0])
a = a[a[7] == 0]
a = a[a[4] == 0]
a = a[a[0] == 0]
a = df_train.iloc[list(set(df_train.index) - set(a.index))]
balanced_df = a.append(a[a[7] == 1])
balanced_df = balanced_df.append(a[a[0] == 1])
show_classes_count(balanced_df)


# For now, 7 classs could have more samples, but generally classes are much more balanced

# # Feature extraction

# ## Inception

# In[30]:


from keras.applications import InceptionV3, ResNet50
from keras.applications.inception_v3 import preprocess_input as inception_preprocess
from keras.applications.resnet50 import preprocess_input as resnet_preprocess
from keras.preprocessing import image
from keras.models import Model, Sequential
from keras.layers import Dense, Flatten
import tensorflow as tf


# In[31]:


from operator import methodcaller


# In[63]:


def get_images(df):
    return list(
        map(lambda x: list(map(get_img, x)),
            map(methodcaller("split", " "), df["photos_id"])))


# In[33]:


df_train["resnet_feature"] = 0 


# In[34]:


len(df_train)


# In[35]:


# imgs = get_images(df_train);


# In[36]:


def modify_model(original_model):
    original_model.layers.pop()
    bottleneck_model = Model(inputs=original_model.inputs, outputs=original_model.layers[-1].output)

    for layer in bottleneck_model.layers:
        layer.trainable = False

    return bottleneck_model


# In[40]:


model = modify_model(ResNet50(weights='imagenet'))

def get_feature(img):
    x = model.predict(resnet_preprocess(np.expand_dims(image.img_to_array(img), axis=0)))
    return x
    
cur_df = df_train
for ind in range(len(cur_df)):
    imgs = get_images(cur_df.iloc[ind:ind+1])[0]
    features = list(map(get_feature, imgs))

    feature = np.array(features).mean(axis=0)
    cur_df.iloc[ind, -1] = feature.flatten().tostring()
    print(ind, len(feature.flatten()))


# In[90]:


with open("result_resnet.csv", "w") as f:
    f.write(cur_df.to_csv())


# In[ ]:


#feat.shape


# In[ ]:


#df_train.to_csv()


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


# In[93]:


import pandas as pd
import numpy as np
#a = pd.read_csv("1.csv")
# eval(a["feature"].values[0])
#a["feature"].apply(lambda x: np.fromstring(eval(x), dtype=np.float32))


# In[ ]:





# In[ ]:


# import numpy as np
# [len(x) for x in a["feature"].values]


# In[ ]:




