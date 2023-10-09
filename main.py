'''Libary needed'''

import numpy as np
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

'''Calsification'''
#calculate the distance between two points
def dis(x1,x2):
  return np.linalg.norm(x1-x2)

#classification
def myclassifier(Train,Trainlabel,Test):
  pred=[]
  for testpoint in Test:
    pred_dis=[]
    for trainpoint in Train:
      pred_dis.append(dis(testpoint,trainpoint))
    pred.append(Trainlabel[np.argmin(pred_dis)])
  return np.array(pred)

'''calculate accuracy'''
def calculate_accuracy(true_labels,predicted_labels):
  if len(true_labels)!=len(predicted_labels):
    raise ValueError("Length of true_label must be the same.")
  #count the number of correct predictions
  correct_predictions=sum(1 for true, predicted in zip(true_labels,predicted_labels)if true == predicted)
  #calculate accuracy as the ratio of correct predictions to total predictions
  accuracy=correct_predictions / len(true_labels)
  return accuracy

# Read File
#read data
Traindata=pd.read_csv('TrainData.csv')
TestData=pd.read_csv('TestData.csv')

Trainlabel=Traindata.iloc[:,-1]#out of all select last col
Testlabel=TestData.iloc[:,-1]

label1=Trainlabel.to_numpy()# classifier
label2=Testlabel.to_numpy()#added accuracy

Trainx=Traindata.iloc[::-1]#select all and start from back to front col
Testx=TestData.iloc[::-1]

Train=Trainx.to_numpy()
Test=Testx.to_numpy()

# label1=Trainlabel.to_numpy()
# label2=Testlabel.to_numpy()

# print(Train)
#*** STOP AND CHECK
#apply pca to train data
k = 110
project_train,reduced_eigenvectors,mean1=my_pca(Train,k) #CHECK AND DOWN ASWELL
#apply my_pca for
Test_centered=Test-mean1
projected_test=np.dot(Test_centered,reduced_eigenvectors)#
#apply classifier on my_pca
predict_label=myclassifier(project_train,label1,projected_test)
accuracy=calculate_accuracy(label2,predict_label)

percnetage_accuracy=accuracy*100
print(f"my PCA accuracy percentage {percnetage_accuracy:.2f} %")