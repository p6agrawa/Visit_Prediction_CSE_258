import gzip
from collections import defaultdict
from random import randint                                                                                                                                                                                                                    
from math import sqrt
from math import fabs
import operator
import os
from datetime import datetime
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

def readGz(f):
  for l in gzip.open(f):
    yield eval(l)

### Would-visit baseline: just rank which businesses are popular and which are not, and return '1' if a business is among the top-ranked

def neg_sampling(user,business,user_buisness):
  global validation_data
  count=0;
  while(count < 50000):
    i = randint(0,len(user)-1)
    j = randint(0,len(business)-1)
    if business[j] in user_buisness[user[i]]:
      continue;
    else:
      validation_data[(user[i],business[j])] = 0
      count+=1

def neg_sampling_train(user,business,user_buisness):
  global train_data
  count=0;
  while(count < 200000):
    i = randint(0,len(user)-1)
    j = randint(0,len(business)-1)
    if business[j] in user_buisness[user[i]]:
      continue;
    else:
      train_data[(user[i],business[j])] = 0
      count+=1


u_b_dict = defaultdict()
user_buisness = defaultdict(set)
businessCount = defaultdict(int)
totalPurchases = 0
user_d = defaultdict()
business_d = defaultdict()

train_data = defaultdict()
validation_data=defaultdict()

counter = 0
for l in readGz("train.json.gz"):
  a,b = l['userID'],l['businessID']
  u_b_dict[a] = 1
  u_b_dict[b] = 1
  user_d[a] =1
  business_d[b] = 1
  if counter < 200000:
    train_data[(a,b)] = 1
    user_buisness[a].add(b)
  else:
    validation_data[(a,b)] = 1
  counter+=1

    

user = []
for k,v in user_d.iteritems():
  user.append(k)
business = []
for k,v in business_d.iteritems():
  business.append(k)

u_b_index = []
for k,v in u_b_dict.iteritems():
  u_b_index.append(k)

for i in range(len(u_b_index)):
  u_b_dict[u_b_index[i]] = i

# neg_sampling(user,business,user_buisness)

# f = open('N2Vtrain_full.txt','w')
# for k,v in train_data.iteritems():
#   a,b = k
#   f.write(str(u_b_dict[a]) + ' ' + str(u_b_dict[b])+'\n')

# f.close()

for k,v in train_data.iteritems():
  a,b = k
  businessCount[b] += 1
  totalPurchases += 1

mostPopular = [(businessCount[x], x) for x in businessCount]
mostPopular.sort()
mostPopular.reverse()

return1 = set()
count = 0
for ic, i in mostPopular:
  count += ic
  return1.add(i)
  if count > (totalPurchases)/2.0: break


neg_sampling_train(user,business,user_buisness)

# 39882 128 
# N2v MODEL

deep1t = np.array(pd.read_csv('Embeddings_8_30_6_1.txt', delimiter="\s+",header=None))
matrixSize = max(deep1t[:,0]).astype(int)+1
deep1 = np.zeros([matrixSize,128])

for i in range(0, deep1t.shape[0]):
    deep1[deep1t[i,0].astype(int)] = deep1t[i,1:129];

print '1st embedding loaded'

X=[]
Y=[]
for k,v in train_data.iteritems() :
  a,b  = k
  X.append(np.multiply(deep1[u_b_dict[a],:],deep1[u_b_dict[b],:]))
  Y.append(v)

print 'data modelled'

X = np.array(X)
Y = np.array(Y)
print X.shape, Y.shape


# A=[]
# B=[]

# for k,v in validation_data.iteritems() :
#   a,b  = k
#   A.append(np.multiply(deep1[u_b_dict[a],:],deep1[u_b_dict[b],:]))
#   B.append(v)

# A = np.array(A)
# B = np.array(B)



model = LogisticRegression( max_iter = 10000, C=0.0001 )
model.fit(X,Y)

# A = np.reshape(A,(A.shape[0],1))
# res = model.predict(A)
# print accuracy_score(B,res)

counter =0
sums =0
new_u = 0
new_b = 0
predictions = open("predictions_Visit_N2V_Full_l30_C8_6_1.txt", 'w')
for l in open("pairs_Visit.txt"):
  if l.startswith("userID"):
    #header
    predictions.write(l)
    continue
  u,i = l.strip().split('-')
  counter +=1
  if  u not in u_b_dict:
    new_u +=1
    predictions.write(u + '-' + i + ",0\n")  
  else:
    if i not in u_b_dict:
      new_b +=1
      predictions.write(u + '-' + i + ",0\n")
    else:
      a = np.multiply(deep1[u_b_dict[u],:],deep1[u_b_dict[i],:])
      a = np.array(a)
      a = np.reshape(a,(1,-1))
      pred = model.predict(a)
      sums += pred[0]
      predictions.write(u + '-' + i + ",{}\n".format(str(pred[0]))) 
predictions.close()

print (1.0*sums)/(1.0*counter)
print new_u,new_b