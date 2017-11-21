import gzip
from collections import defaultdict
from random import randint
from math import sqrt
from math import fabs
import operator

def readGz(f):
  for l in gzip.open(f):
    yield eval(l)

### Would-visit baseline: just rank which businesses are popular and which are not, and return '1' if a business is among the top-ranked

def neg_sampling(user,business,user_buisness):
  global validation_data
  count=0;
  while(count < 100000):
    i = randint(0,len(user)-1)
    j = randint(0,len(business)-1)
    if business[j] in user_buisness[user[i]]:
      continue;
    else:
      validation_data[(user[i],business[j])] = 0
      count+=1


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
  user_d[a]=1
  business_d[b]=1
  if counter < 100000:
    train_data[(a,b)] = 1
  else:
    validation_data[(a,b)] = 1
  counter+=1
  user_buisness[a].add(b)
    
user=[]
for k,v in user_d.iteritems():
  user.append(k)
business=[]
for k,v in business_d.iteritems():
  business.append(k)

for k,v in train_data.iteritems():
  a,b = k
  businessCount[b] += 1
  totalPurchases += 1

mostPopular = [(businessCount[x], x) for x in businessCount]
mostPopular.sort()
mostPopular.reverse()

neg_sampling(user,business,user_buisness)

for x in range(1,10):
  return1 = set()
  count = 0
  for ic, i in mostPopular:
    count += ic
    return1.add(i)
    if count > (x*totalPurchases)/10.0: break

  counter=0
  prediction = []

  for k,v in validation_data.iteritems():
    u,b = k
    if b in return1:
      prediction.append(1)
    else:
      prediction.append(0)
  sums =0.0
  for a,b in zip(prediction,validation_data.values()):
    sums+= (a==b)

  print 'Accuracy of Baseline Model(threshold = {}%) on the  Validation Set : {} '.format(x*10,sums/(1.0*len(validation_data)))

# UPDATED MODEL

user_cat = defaultdict(set)
business_cat = defaultdict(set)
counter = 0
for l in readGz("train.json.gz"):
  a,b,c = l['userID'],l['businessID'],l['categories']
  if(counter >= 100000):
    break
  for x in c:
    user_cat[a].add(x)
    business_cat[b].add(x)
  counter+=1

print len(user_cat.keys()),len(business_cat.keys())

new_prediction=[]
for k,v in validation_data.iteritems():
  u,b = k
  pred = False
  for x in business_cat[b]:
    if x in user_cat[u]:
      pred = True
      break
  if pred:
    new_prediction.append(1)
  else:
    new_prediction.append(0)

sums =0.0
for a,b in zip(new_prediction,validation_data.values()):
  sums += (a==b)
print 'Accuracy of Updated Model on the  Validation Set : {} '.format(sums/(1.0*len(validation_data)))


predictions = open("predictions_Visit.txt", 'w')
for l in open("pairs_Visit.txt"):
  if l.startswith("userID"):
    #header
    predictions.write(l)
    continue
  u,i = l.strip().split('-')
  pred = False
  for x in business_cat[i]:
    if x in user_cat[u]:
      pred = True
      break
  if pred:
    predictions.write(u + '-' + i + ",1\n")
  else:
    predictions.write(u + '-' + i + ",0\n")
predictions.close()

### Rating baseline: compute averages for each user, or return
# the global average if we've never seen the user before
# Part 5,6,7,8 

allRatings = []
userRatings = defaultdict(list)
for l in readGz("train.json.gz"):
  user,business = l['userID'],l['businessID']
  allRatings.append(l['rating'])
  userRatings[user].append(l['rating'])

globalAverage = sum(allRatings[:len(allRatings)/2]) / (len(allRatings)/2)

userAverage = {}
for u in userRatings:
  userAverage[u] = sum(userRatings[u]) / len(userRatings[u])

test = allRatings[len(allRatings)/2:]
RMSE = 0.0

for t in test:
  RMSE += (t - globalAverage)**2

RMSE /= len(test)
RMSE = sqrt(RMSE)

print 'RMSE on the Validation Set is : {}'.format(RMSE)

# Part 6 - New Model
user_dict = defaultdict(set)
business_dict = defaultdict(set)
user_buisness_rating = defaultdict(dict)
user =[]
business=[]
for l in readGz("train.json.gz"):
  a,b = l['userID'],l['businessID']
  user.append(a)
  business.append(b)
  user_buisness_rating[a][b] = l['rating']

for i in range(len(user)/2):
  a = user[i]
  b = business[i]
  user_dict[a].add(b)
  business_dict[b].add(a)
  
bu = defaultdict(float)
bi = defaultdict(float)
a = 0.0
lam = 1.0

def loss(R):
  global a
  global bu
  global bi
  loss =0.0
  for i in range(len(user)/2,len(user)):
    x = user[i]
    y = business[i]
    loss += (a + bu[x] + bi[y] - R[x][y])**2

  loss /= 1.0*(len(user)-len(user)/2)
  loss = sqrt(loss)
  return loss

def training_loss(R):
  global a
  global bu
  global bi
  loss =0.0
  for i in range(len(user)/2):
    x = user[i]
    y = business[i]
    loss += (a + bu[x] + bi[y] - R[x][y])**2

  loss /= 1.0*(len(user)/2)
  loss = sqrt(loss)
  return loss


def update(R,lam):
  global a
  global bu
  global bi
  RMSE = training_loss(R)
  prev_RMSE = 5.0
  counter = 0
  while ( fabs(prev_RMSE - RMSE) > 0.0001 ):
    temp = 0.0
    for i in range(len(user)/2):
      x = user[i]
      y = business[i]
      temp += 1.0*(R[x][y] - (bu[x] +bi[y]))
    temp /= (1.0*len(user))/2
    a = temp

    for u,l in user_dict.iteritems():
      temp = 0.0
      for i in l:
        temp += 1.0*(R[u][i] - (a +bi[i]))
      
      temp /= 1.0*(lam+len(l))
      bu[u] = temp

    for b,l in business_dict.iteritems():
      temp = 0.0
      for i in l:
        temp += 1.0*(R[i][b] - (a +bu[i]))
      temp /= 1.0*(lam+len(l))
      bi[b] = temp
    prev_RMSE =RMSE
    RMSE = training_loss(R)
    counter+=1
    # print 'RMSE at convergence step {} Training : {}, Validation : {}'.format(counter,RMSE,loss(R))
    # print 'Aplha = {} '.format(a)

  print '\nRMSE at convergence. Lambda = {} Training : {}, Validation : {}\n'.format(lam,RMSE,loss(R))


update(user_buisness_rating,lam)

sorted_BU = sorted(bu.iteritems(), key=operator.itemgetter(1))
sorted_BI = sorted(bi.iteritems(), key=operator.itemgetter(1))

print 'Alpha = {}\nTen highest B_u users = {}\nTen highest B_i business = {}\n'.format(a,sorted_BU[-10:],sorted_BI[-10:])

print 'User with smallest B_u is {}, and largest B_u is {}'.format(sorted_BU[0],sorted_BU[-1])
print 'User with smallest B_i is {}, and largest B_i is {}'.format(sorted_BI[0],sorted_BI[-1])

lam = [0.01,0.1,1,10,100]
for l in lam:
  update(user_buisness_rating,l)

update(user_buisness_rating,10)

predictions = open("predictions_Rating.txt", 'w')
for l in open("pairs_Rating.txt"):
  if l.startswith("userID"):
    #header
    predictions.write(l)
    continue
  u,i = l.strip().split('-')
  prediction = a + bu[u] + bi[i]
  predictions.write(u + '-' + i + ',' + str(prediction) + '\n')
  
predictions.close()
