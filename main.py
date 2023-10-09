'''Libary needed'''
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA
# from sklearn.decomposition import KernalPCA

#pca function
# from numpy import einsum
def my_pca(Data,k):
  #calculate mean of the input along each col(featuere)
  mean1=np.mean(Data,axis=0)

  #center data
  D1=Data-mean1

  #covaraince matrix of centered data
  cov=np.cov(D1,rowvar=False)

  #eigenvalues and eigenvectors of covaraince matrix
  eigenvalues,eigenvectors=np.linalg.eig(cov)

  #sort eigenvalues in descending order and rearrange those eigenvectors
  idx=np.argsort(eigenvalues)[::-1]
  eigenvalues=eigenvalues[idx]
  eigenvectors=eigenvectors[:,idx]

  #select top eigenvectors and reduce the dimensionalty
  reduced_eigenvectors=eigenvectors[:,:k]

  #project the centered data to the reduced eignvectors
  projecteddata=np.dot(D1,reduced_eigenvectors)

  return projecteddata,reduced_eigenvectors,mean1
