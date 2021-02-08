#!/usr/bin/env python
# coding: utf-8

# In[27]:


import numpy as np
from numba import njit
from functionality import *
import matplotlib.pyplot as plt
from skimage.io import imread
import time
from PIL import Image


# In[28]:


imleft = np.array(Image.open('test_images/im0.ppm').convert('L')).astype(float)#imread('test_images/im1.png',as_gray=True).astype("float")
imright = np.array(Image.open('test_images/im1.ppm').convert('L')).astype(float)#imread('test_images/im2.png',as_gray=True).astype("float")
height,width = imleft.shape


# In[29]:


mindx,maxdx = 0,20
mindy,maxdy = 0,0
smooth = 1


# In[30]:


disp_x = np.arange(mindx, maxdx+1, 1)
disp_y = np.arange(mindy,maxdy+1, 1)
X2D,Y2D = np.meshgrid(disp_y,disp_x)
disparity_vec = np.column_stack((X2D.ravel(),Y2D.ravel()))


# In[31]:


n_labels = disparity_vec.shape[0]
n_labels


# In[32]:


Q = np.full((height,width,n_labels),-np.inf)
for i in range(height):
    for j in range(width):
        for index, d in enumerate(disparity_vec):
            if 0 <= i-d[0] < height and 0 <= j-d[1] < width:
                Q[i,j,index] = -abs(imleft[i,j]-imright[i-d[0],j-d[1]])


# In[33]:


def calc_g(d1,d2,mapping):
    vec1 = mapping[d1]
    vec2 = mapping[d2]
    return np.linalg.norm(vec1-vec2)


# In[34]:


g = np.full((n_labels,n_labels),-1)
for i in range(n_labels):
    for j in range(n_labels):
        g[i,j] = calc_g(i,j,disparity_vec)
g = -g*smooth


# In[35]:


@njit(fastmath=True, cache=True)
def init(height,width,n_labels,Q,g,P):
    # for each pixel of input channel
    # going from bottom-right to top-left pixel
    for i in np.arange(height-2,-1,-1):
        for j in np.arange(width-2,-1,-1):
            # for each label in pixel
            for k in range(n_labels):
                # P[i,j,1,k] - Right direction
                # P[i,j,3,k] - Down direction
                # calculate best path weight according to formula
                P[i,j,3,k] = max(P[i+1,j,3,:] + Q[i+1,j,:] + g[k,:])
                P[i,j,1,k] = max(P[i,j+1,1,:] + Q[i,j+1,:] + g[k,:])
    return P


# In[36]:


@njit(fastmath=True, cache=True)
def forward_pass(height,width,n_labels,Q,g,P):
    # for each pixel of input channel
    for i in range(1,height):
        for j in range(1,width):
            # for each label in pixel
            for k in range(n_labels):
                # P[i,j,0,k] - Left direction
                # P[i,j,2,k] - Up direction
                # calculate best path weight according to formula
                P[i,j,0,k] = max(P[i,j-1,0,:] + Q[i,j-1,:] + g[:,k])
                P[i,j,2,k] = max(P[i-1,j,2,:] + Q[i-1,j,:] + g[:,k])
            
    return P


# In[45]:


P = np.zeros((height,width,4,n_labels))


# In[46]:


P = init(height,width,n_labels,Q,g,P)


# In[47]:


P = forward_pass(height,width,n_labels,Q,g,P)


# In[48]:


optimal_labelling = np.argmax(P[:,:,0,:] + P[:,:,1,:] + P[:,:,2,:] + P[:,:,3,:] + Q,axis=2)


# In[49]:


plt.figure(figsize=(20,20))
plt.axis('off')
plt.imshow(optimal_labelling,cmap='gray')


# In[42]:


len(optimal_labelling[optimal_labelling==0])


# In[43]:


pic = np.linalg.norm(disparity_vec[optimal_labelling],axis=2)


# In[44]:


plt.figure(figsize=(20,20))
plt.axis('off')
plt.imshow(pic,cmap='gray')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




