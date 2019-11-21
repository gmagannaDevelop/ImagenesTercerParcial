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
from itertools import chain

# Visualisation : 
import matplotlib.pyplot as plt
import matplotlib.image as pim
import matplotlib.patches as mpatches
import seaborn as sns

# Data tools :
import numpy as np
import pandas as pd

# Image processing : 
import cv2 as cv
from skimage import data
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb

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


# In[75]:


# Being lazy :
lmap = lambda x, y: list(map(x, y))
lfilter = lambda x, y: list(filter(x, y))


# In[3]:


#plt.style.available


# In[4]:


plt.style.use('seaborn-deep')
plt.rcParams['figure.figsize'] = (10, 5)


# In[5]:


img   = cv.imread('imagenes/Ex3Preg6(a).tif', cv.IMREAD_GRAYSCALE)
color = cv.cvtColor(img, cv.COLOR_GRAY2RGB) # Color copy, to draw colored circles


# ## a. (0.5 puntos) Usando una técnica de umbralización global, segmente la imagen y muestre el resultado de la segmetnación.

# In[6]:


intensities = pd.core.frame.DataFrame(dict(intensity=img.flatten()))


# In[7]:


sns.distplot(intensities, kde=False, rug=True, bins=10)


# In[69]:


kmeans = KMeans(n_clusters=2, random_state=0, verbose=False).fit(intensities)
K = int(kmeans.cluster_centers_.mean())


# In[99]:


centers1 = lmap(int, list(chain.from_iterable(kmeans.cluster_centers_)))


# In[74]:


sns.distplot(intensities, kde=False, rug=True, bins=10)
plt.axvline(K, color='r')
lmap(lambda x: plt.axvline(x, color='g'), centers1)
_ = plt.title(f"Means = {centers1}, K = {K}", size=16)


# In[10]:


thresh1 = cv.threshold(img, K, 255, cv.THRESH_BINARY)[1]


# In[11]:


utils.side_by_side(img, thresh1)


# Como podemos ver, una técinca de umbralización estándar como k-medias móviles, con dos medias, da resultados muy pobres.

# In[12]:


otsu1 = cv.threshold(img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)[1]


# In[13]:


utils.side_by_side(img, otsu1)


# El algoritmo de Otsu no logra mejorar mucho la segmentación (esto era de esperarse dado que el histograma original era claramente bimodal). 

# In[14]:


gblur = cv.GaussianBlur(img,(3,3),0)
otsu2 = cv.threshold(gblur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)[1]


# In[15]:


utils.side_by_side(img, otsu2)


# El algoritmo de Otsu no logra mejorar mucho la segmentación aún en combinación con un suavizado Gaussiano.

# In[16]:


img_blur = cv.medianBlur(img, 5)
circles = cv.HoughCircles(img_blur, cv.HOUGH_GRADIENT, 1, img.shape[0]/64, param1=200, param2=10, minRadius=5, maxRadius=7)


# In[17]:


if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        cv.circle(img, (i[0], i[1]), i[2], (0, 0, 255), 2)


# In[18]:


plt.imshow(img, cmap='gray')


# Al ver que las células tenían una apariencia más o menos circular, parecía una buena idea usar una transformada de Hough para buscar los círculos de la imagen, pero esto entregó resultados muy pobres. Tal vez esto podría funcionar con la imagen binaria.

# ## b. (0.5 puntos) A la imagenoriginal se le aplicó una umbralización con valores locales yal resultado se le realizó una apertura morbológica obteniendo la imagen Ex3Preg6(b).tif. Usando esta imagen,cuente y etiquete cuantos objetos de la segmentación pueden considerarse células independientes. 

# In[19]:


imgb  = cv.imread('imagenes/Ex3Preg6(b).tif', cv.IMREAD_GRAYSCALE)
imgbc = cv.cvtColor(imgb, cv.COLOR_BAYER_GB2RGB)
utils.side_by_side(imgb, imgbc)


# ### Primera aproximación :
# El uso de la transformada de Hough para círculos, no para líneas. Al ver la imagen, uno podría pensar que un círculo es una buena aproximación de la forma de una célula, por lo tanto los cículos encontrados por una transformada de Hough serían las células que buscamos identificar, caracterizar y contabilizar

# In[20]:


circles = cv.HoughCircles(imgb, cv.HOUGH_GRADIENT, 1, img.shape[0]/64, param1=200, param2=10, minRadius=5, maxRadius=15)


# In[21]:


if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        cv.circle(imgbc, (i[0], i[1]), i[2], (155, 0, 0), 2)


# In[22]:


utils.side_by_side(imgb, imgbc)


# Aquí podemos ver que aunque la imagen principal carece de falsos positivos (es decir todos los círculos dibujados dentro de la región útil de la imagen contienen una célula) el **número de falsos negativos es altísimo** : sólo una pequeña parte de las células observadas fueron identificadas por ```cv.HoughCircles()```. 
# 
# Esto nos indica que tal vez las células no se asemejan tanto a un círculo. Por esta razón, no se explorará más a fondo esta vía de acción. Cabe mencionar que la transformada encuentra círculos en el texto de encabezado : **Imagen con apertura**. Por esta razón, en delante se trabajará con otra imagen recortada a mano para excluir este texto que podría causar problemas en la segmentación más adelante.

# In[23]:


imgb2  = cv.imread('imagenes/Ex3Preg6(b)3.tif', cv.IMREAD_GRAYSCALE)
imgb2c = clear_border(imgb2)
utils.side_by_side(imgb2, imgb2c)


# In[24]:


sns.distplot(imgb2c.flatten(), kde=False, rug=True)


# Imagen claramente binaria.

# In[25]:


label_image, n_objs = label(imgb2c, return_num=True)
fig, ax = plt.subplots(figsize=(10, 10))
#ax.imshow(label_image[100:200,0:100:])
ax.imshow(label_image)
print(n_objs)


# In[26]:


objs = regionprops(label_image) 
areas = pd.core.frame.DataFrame({
    'area': map(lambda x: x.area, objs)
})


# In[28]:


areas.describe(), sns.boxplot(areas)


# In[30]:


sns.distplot(areas, kde=False, rug=True)


# In[41]:


_ = areas.area.sort_values()
_[:10], _[-10:]


# Algunos de los primeros valores de las áreas corresponden claramente a falsos positivos, porque no tenemos células de uno o dos pixeles. Estos pixeles blancos no se observaban en la imagen original, de cualquier manera se retirarán manualmente para no sesgar el análisis posterior.

# In[44]:


objs2  = regionprops(label_image)
objs2  = list(filter(lambda x: x if x.area > 2 else False, objs2))
areas2 = pd.core.frame.DataFrame({
    'area': map(lambda x: x.area, objs2)
})


# In[98]:


kmeans2 = KMeans(n_clusters=3, random_state=0, verbose=False).fit(areas2)
centers = pd.core.frame.DataFrame({
    "means": chain.from_iterable(kmeans2.cluster_centers_)
})
centers['k'] = centers.rolling(2).mean()
print(centers)
centers = centers.applymap(lambda x: np.int64(x) if not np.isnan(x) else x)
print(centers)


# In[61]:


sns.distplot(areas2, kde=False, rug=True)
lmap(lambda x: plt.axvline(x, color='r'), centers.k.dropna())
lmap(lambda x: plt.axvline(x, color='g'), centers.means)
_ = plt.title(f"Means = {lmap(int, centers.means.tolist())}, K = {lmap(int, centers.k.dropna().tolist())}", size=16)


# In[100]:


ks = centers.k.dropna().tolist()
cells = [
    lfilter(lambda x: x if x.area < ks[0] else False, objs2),
    lfilter(lambda x: x if x.area >= ks[0]  and x.area < ks[1] else False, objs2),
    lfilter(lambda x: x if x.area >= ks[1] else False, objs2)
]


# In[54]:


fig, ax = plt.subplots(figsize=(9, 9))
ax.imshow(imgb2c, cmap='gray')

for region in objs2:
    # take regions with large enough areas
    if region.area < centers.k.dropna().tolist()[0]:
        # draw rectangle around segmented cells
        minr, minc, maxr, maxc = region.bbox
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)

#ax.set_axis_off()
plt.tight_layout()
plt.show()


# In[50]:


fig, ax = plt.subplots(figsize=(9, 9))
ax.imshow(imgb2c, cmap='gray')
ks = centers.k.dropna().tolist()

for region in objs:
    # take regions with large enough areas
    if region.area > ks[0] and region.area <= ks[1]:
        # draw rectangle around segmented cells
        minr, minc, maxr, maxc = region.bbox
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)

#ax.set_axis_off()
plt.tight_layout()
plt.show()


# In[51]:


fig, ax = plt.subplots(figsize=(9, 9))
ax.imshow(imgb2c, cmap='gray')
ks = centers.k.dropna().tolist()

for region in objs:
    # take regions with large enough areas
    if region.area > ks[1]:
        # draw rectangle around segmented cells
        minr, minc, maxr, maxc = region.bbox
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)

#ax.set_axis_off()
plt.tight_layout()
plt.show()


# # Extra

# In[53]:


filter(lambda x: x, [True, False])


# In[55]:


range(ks[0], ks[1])


# In[76]:


plt.close('all')


# In[80]:


(lambda x, y: x(*y))(print, [1, 2, 3, ""])


# In[89]:


help(pd.core.frame.DataFrame.lapply)


# In[ ]:




