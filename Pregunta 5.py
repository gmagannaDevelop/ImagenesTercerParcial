#!/usr/bin/env python
# coding: utf-8

# # Pregunta 5

# (3 puntos) Con análisis morfológico analice la imagen llamada Ex3Preg5.tif
# 
# a. (1 punto) Haga una función que automáticamente cuente cuantas letras hay en la imagen.
# 
# b. (1 punto) Haga una función que automáticamente cuentecuantas letras mayúsiculas hay en la imagen, la función debe regresar también una tabla cuyas columnas indiquien el renglón y el número de letra en el que aparece la mayúsculaen dicho renglón, habrá una fila de esta tabla por cada mayúsucula encontrada.
# 
# c. (1 punto) Haga una función que automáticamente cuentecuantas letras “o”minúsculas hay en la imagen.La función debe regresar también una tabla cuyas columnas indiquien el renglón y el número de letra en el que aparece la “o”en dicho renglón, habrá una fila de esta tabla por cada “o”encontrada

# In[2]:


from functools import partial, reduce

import matplotlib.pyplot as plt
import matplotlib.image as img

import numpy as np
import cv2 as cv


# In[ ]:




