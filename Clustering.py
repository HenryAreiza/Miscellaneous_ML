#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 25, 2023

@author: HenRick

This code uses scikit-learn library to apply unsupervised learning (clustering)
over a set of images, it uses statistical features generated from two different
color spaces

"""

###############################################################################
###############################################################################
import os
import numpy as np
from skimage import io
from skimage.color import rgb2hsv
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import random


path = "./Data/"
subfolders = ["Fold1", "Fold2", "Fold3"]

features_all1=[]
features_all2=[]
images = []
names = []
for i in subfolders:
    subsubfolders = os.listdir(os.path.join(path, i))
    for j in subsubfolders:
        p = os.path.join(path, i, j)
        file = os.listdir(p)

        for ind,t in enumerate(file):
            print(f"{ind}")
            names.append(i + "_" + t[-14:-4])
            features_c = []
            features_c2 = []
            image = io.imread(os.path.join(p,t))
            red_channel = image[:, :, 0]
            green_channel = image[:, :, 1]
            blue_channel = image[:, :, 2]
                    
            # Calculate features for each channel RGB space
            features_c.append(np.mean(red_channel))
            features_c.append(np.mean(green_channel))
            features_c.append(np.mean(blue_channel))
            
            features_c.append(np.std(red_channel))
            features_c.append(np.std(green_channel))
            features_c.append(np.std(blue_channel))
            
            features_c.append(np.max(red_channel))
            features_c.append(np.max(green_channel))
            features_c.append(np.max(blue_channel))
            
            features_c.append(np.min(red_channel))
            features_c.append(np.min(green_channel))
            features_c.append(np.min(blue_channel))
            
            # Calculate features for each channel HSV space 
            img_hsv = rgb2hsv(image)
            h_channel = img_hsv[:, :, 0]
            s_channel = img_hsv[:, :, 1]
            v_channel = img_hsv[:, :, 2]
            features_c2.append(np.mean(h_channel))
            features_c2.append(np.mean(s_channel))
            features_c2.append(np.mean(v_channel))
            
            features_c2.append(np.std(h_channel))
            features_c2.append(np.std(s_channel))
            features_c2.append(np.std(v_channel))
            
            features_c2.append(np.max(h_channel))
            features_c2.append(np.max(s_channel))
            features_c2.append(np.max(v_channel))
            
            features_c2.append(np.min(h_channel))
            features_c2.append(np.min(s_channel))
            features_c2.append(np.min(v_channel))

            features_all1.append(features_c2)
            features_all2.append(features_c+features_c2)
            images.append(image)
            

# Estandarizar las características
scaler = StandardScaler()
features_all1 = scaler.fit_transform(features_all1)
features_all2 = scaler.fit_transform(features_all2)


# Realizar clustering con KMeans
n_clusters = 4  # Establece el número de clusters que deseas

kmeans1 = KMeans(n_clusters=n_clusters, random_state=0).fit(features_all1)
labels1 = kmeans1.predict(features_all1)
silhouette_avg = silhouette_score(features_all1, labels1)
print("silhouette_avg kmeans HSV: ", silhouette_avg)

kmeans2 = KMeans(n_clusters=n_clusters, random_state=0).fit(features_all2)
labels2 = kmeans2.predict(features_all2)
silhouette_avg = silhouette_score(features_all2, labels2)
print("silhouette_avg kmeans RGB + HSV: ", silhouette_avg)


# Reducción de dimensión
pca = PCA(n_components=2)

images_pca1 = pca.fit_transform(features_all1)
X_embedded1 = TSNE(n_components=2, learning_rate='auto',
                  init='random', perplexity=5,random_state=42).fit_transform(features_all1)

images_pca2 = pca.fit_transform(features_all2)
X_embedded2 = TSNE(n_components=2, learning_rate='auto',
                  init='random', perplexity=5,random_state=42).fit_transform(features_all2)


# Plot Results
plt.subplot(2, 2, 1)
plt.scatter(images_pca1[:, 0], images_pca1[:, 1], c=labels1)
plt.title('kmeans HSV + PCA')

plt.subplot(2, 2, 2)
plt.scatter(X_embedded1[:, 0], X_embedded1[:, 1], c=labels1)
plt.title('kmeans HSV + TSNE')

plt.subplot(2, 2, 3)
plt.scatter(images_pca2[:, 0], images_pca2[:, 1], c=labels2)
plt.title('kmeans RGB-HSV + PCA')

plt.subplot(2, 2, 4)
plt.scatter(X_embedded2[:, 0], X_embedded2[:, 1], c=labels2)
plt.title('kmeans RGB-HSV + TSNE')
plt.show()


# Visualizar algunas imágenes
Im2Plot = 10
lim = [len(np.where(labels1==label)[0]) for label in range(n_clusters)]
SelImg = random.sample(range(min(lim)), Im2Plot)
plt.figure()
for label in range(n_clusters):
    ind = np.where(labels1==label)[0]
    for im in range(Im2Plot):
        plt.subplot(n_clusters, Im2Plot, (Im2Plot*label) + im + 1)
        plt.imshow(images[ind[SelImg[im]]])
        plt.axis('off')
        if im == 0:
            plt.title(f'HSV - Cluster {label + 1}\n{names[ind[SelImg[im]]][:10]}')
        else:
            plt.title(f'{names[ind[SelImg[im]]][:10]}')
plt.show()

lim = [len(np.where(labels2==label)[0]) for label in range(n_clusters)]
SelImg = random.sample(range(min(lim)), Im2Plot)
plt.figure()
for label in range(n_clusters):
    ind = np.where(labels2==label)[0]
    for im in range(Im2Plot):
        plt.subplot(n_clusters, Im2Plot, (Im2Plot*label) + im + 1)
        plt.imshow(images[ind[SelImg[im]]])
        plt.axis('off')
        if im == 0:
            plt.title(f'RGB + HSV - Cluster {label + 1}\n{names[ind[SelImg[im]]][:10]}')
        else:
            plt.title(f'{names[ind[SelImg[im]]][:10]}')
plt.show()

#%%

# Visualizar más imágenes!!

print([names[x] for x in np.where(labels2==1)[0]])

SelImg = random.sample(range(480), 50)
plt.figure()
ind = np.where(labels2==1)[0]
for im in range(50):
    plt.subplot(5, 10, im + 1)
    plt.imshow(images[ind[SelImg[im]]])
    plt.title(f'{names[ind[SelImg[im]]][:8]}')
    plt.axis('off')
plt.show()
