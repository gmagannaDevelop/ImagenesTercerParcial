#!/usr/bin/env python
# coding: utf-8

# # Pregunta 6

# (2 puntos) La imagen Ex3Preg6(a).tif muestra una imagen tomada con un microscopio de cultivo de bacterias identificadas por los círculos intensos:
# 
# a. (0.5 puntos) Usando una técnica de umbralización global, segmente la imagen y muestre el resultado de la segmetnación.
# 
# b. (0.5 puntos) A la imagenoriginal se le aplicó una umbralización con valores locales yal resultado se le realizó una apertura morbológica obteniendo la imagen Ex3Preg6(b).tif. Usando esta imagen,cuente y etiquete cuantos objetos de la segmentación pueden considerarse células independientes. 
# 
# c. (1 punto) Continuando con la imagen anterior. Cuente y etiquete cuantos objetos de la segmentación pueden considerarse 2 células agrupadas, y cuantos y cuales más de 2 células.

# In[1]:


# Functional programing tools : 
from functools import partial, reduce

# Visualisation : 
import matplotlib.pyplot as plt
import matplotlib.image as pim
import seaborn as sns

# Data tools :
import numpy as np
import pandas as pd

# Image processing : 
import cv2 as cv

# Machine Learning :
from sklearn.cluster import KMeans

# Jupyter reimport utils :
import importlib


# In[2]:


# Custom :
import mfilt_funcs as mfs
importlib.reload(mfs)
import mfilt_funcs as mfs


import utils
importlib.reload(utils)
import utils


# In[3]:


#plt.style.available


# In[39]:


plt.style.use('seaborn-deep')
plt.rcParams['figure.figsize'] = (10, 5)


# In[77]:


img     = cv.imread('imagenes/Ex3Preg6(a).tif', cv.IMREAD_GRAYSCALE)
color   = cv.cvtColor(img, cv.COLOR_GRAY2RGB) # Color copy, to draw colored circles


# a. (0.5 puntos) Usando una técnica de umbralización global, segmente la imagen y muestre el resultado de la segmetnación.

# In[41]:


intensities = pd.core.frame.DataFrame(dict(intensity=gthresh.flatten()))


# In[42]:


intensities.hist()


# In[43]:


kmeans = KMeans(n_clusters=2, random_state=0, verbose=False).fit(intensities)
K = kmeans.cluster_centers_.mean()


# In[74]:


intensities.hist()
plt.axvline(K, color='r')
list(map(lambda x: plt.axvline(x, color='g'), kmeans.cluster_centers_))
_ = plt.title(f"Means = {kmeans.cluster_centers_.tolist()}, K = {K}", size=16)


# In[89]:


gthresh = img.copy()
gthresh[ gthresh >= K ] = 255
gthresh[ gthresh < K ] = 0 


# In[90]:


utils.side_by_side(img, gthresh)


# In[86]:


ret, thresh1 = cv.threshold(img, K, 255, cv.THRESH_BINARY)


# In[87]:


utils.side_by_side(img, thresh1)


# Como podemos ver, una técinca de umbralización estándar como k-medias móviles, con dos medias, da resultados muy pobres.

# In[24]:


# Blur the image to reduce noise
img_blur = cv.medianBlur(img, 5)


# In[34]:


# Apply hough transform on the image
circles = cv.HoughCircles(img_blur, cv.HOUGH_GRADIENT, 1, img.shape[0]/64, param1=200, param2=10, minRadius=5, maxRadius=7)


# In[35]:


# Draw detected circles
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # Draw outer circle
        cv.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # Draw inner circle
        #cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)


# In[36]:


side_by_side(img, color)


# In[28]:


help(cv2.HoughCircles)


# In[26]:


help(cv2.cvtColor)


# In[ ]:





# In[ ]:




