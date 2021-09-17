#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt #drawing figures
from sklearn.decomposition import PCA #reduces the dimensionality of a large data set
from sklearn.metrics import classification_report#print the result of prediction
from sklearn.model_selection import train_test_split #split dataset into train/test
from sklearn.svm import SVC #Support Vector Classification is based on libsvm.

#set functions help machine read the images
def show_images(pixels):
    #Display images
    fig,axes = plt.subplots(6,10, figsize=(11,7), subplot_kw={'xticks':[], 'yticks':[]})
    #design the format of print-list/figures
    for i, ax in enumerate(axes.flat):#for-loop able to print images in order
        ax.imshow(np.array(pixels)[i].reshape(64,64), cmap='gray')
        #design image format. image size = 64x64 color=gray.
    plt.show()

def show_eigenfaces(pca):
    #Display eigenfaces
    fig,axes = plt.subplots(3,8, figsize=(9,4), subplot_kw={'xticks':[], 'yticks':[]})
    #design the format of print-list/figures
    for i, ax in enumerate(axes.flat):#for-loop able to print images in order
        ax.imshow(np.array(pixels)[i].reshape(64,64), cmap='gray')
        #design image format. image size = 64x64 color=gray.
    plt.show()

#Step 1: read and load datasets
datasets = pd.read_csv("https://raw.githubusercontent.com/codeheroku/Introduction-to-Machine-Learning/master/Face%20Recognition%20Using%20PCA/face_data.csv")
#load datasets
labels = datasets["target"]
pixels = datasets.drop(["target"], axis=1)

show_images(pixels)
#display images

#Step 2: split dataset into training and testing
x_train, x_test, y_train, y_test = train_test_split(pixels, labels)

#Step 3: perform PCA
pca = PCA(n_components=200).fit(x_train) #find out components on each images
plt.plot(np.cumsum(pca.explained_variance_ratio_))
# explained_ variance_ ratio_ Show the percentage of variance explained 
# by each of the selected components.
# np.cumsum give us a cumulate sum of how much radiation is getting catcher, 
# as we keep increasing the number of components
plt.show()
show_eigenfaces(pca)
#diplay the selected images

#Step 4 Project Training data to PCA
x_train_pca = pca.transform(x_train)
#we want to predict the data under the pca, so 
#We created a PCA object named pca in the previous function. 
#We use transform, the components or features will be returned to the PCA training. 
#Step 5 Make prediction on training dataset
cfit = SVC(gamma='auto')
cfit = cfit.fit(x_train_pca, y_train)

#Step 6 Perform testing and get classification report
x_test_pca = pca.transform(x_test)
y_pred = cfit.predict(x_test_pca)

print (classification_report(y_test, y_pred))#evaluating and predicting


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt #drawing figures
from sklearn.decomposition import PCA #reduces the dimensionality of a large data set
from sklearn.metrics import classification_report#print the result of prediction
from sklearn.model_selection import train_test_split #split dataset into train/test
from sklearn.svm import SVC #Support Vector Classification is based on libsvm.

#set functions help machine read the images
def show_images(pixels):
    #Display images
    fig,axes = plt.subplots(6,10, figsize=(11,7), subplot_kw={'xticks':[], 'yticks':[]})
    #design the format of print-list/figures
    for i, ax in enumerate(axes.flat):#for-loop able to print images in order
        ax.imshow(np.array(pixels)[i].reshape(64,64), cmap='gray')
        #design image format. image size = 64x64 color=gray.
    plt.show()

def show_eigenfaces(pca):
    #Display eigenfaces
    fig,axes = plt.subplots(3,8, figsize=(9,4), subplot_kw={'xticks':[], 'yticks':[]})
    #design the format of print-list/figures
    for i, ax in enumerate(axes.flat):#for-loop able to print images in order
        ax.imshow(np.array(pixels)[i].reshape(64,64), cmap='gray')
        #design image format. image size = 64x64 color=gray.
    plt.show()

#Step 1: read and load datasets
datasets = pd.read_csv("https://raw.githubusercontent.com/codeheroku/Introduction-to-Machine-Learning/master/Face%20Recognition%20Using%20PCA/face_data.csv")
print (datasets.head())
#load datasets


# In[ ]:





# In[ ]:




