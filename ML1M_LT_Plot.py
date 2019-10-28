import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt

data = np.genfromtxt('.../Ratings.dat',delimiter='::')

uID = data[:,][:,0]     # first column
mID = data[:,][:,1]     # second column
rating = data[:,][:,2]  #third column
time = data[:,][:,3]
rating[np.isnan(rating)] = 0

### number of rows(users), columns(movies) and variety of ratings given by users
MIN_uID = int(min(uID))
MAX_uID = int(max(uID))

MIN_mID = int(min(mID))
MAX_mID = int(max(mID))

MIN_rating = min(rating)
MAX_rating = max(rating)

print ('Num of Users: ', MIN_uID, ' to ', MAX_uID, '\nNum of Movies: ', MIN_mID, ' to ', MAX_mID,
       '\nRange of ratings given by users: ', MIN_rating, ' to ', MAX_rating)
##### Raing Matrix

Rating_Matrix = np.zeros((int(MAX_uID), int(max(mID))), float)
# Rating_Matrix[:][:]=np.nan

i = iter(data)
for j in range(len(data)):
    n = next(i)
    x = int(n[0])
    y = int(n[1])
    Rating_Matrix[x - 1][y - 1] = n[2]

print('\n*** the Rating_Matrix is:***\n', Rating_Matrix)

# number of times each item has been rated
a=np.count_nonzero(Rating_Matrix, axis=0)
b=np.sort(a)[::-1]

# plot: the share of long-tail items in the MovieLens dataset
c=range(MAX_mID)
_ = plt.bar(c,b)
plt.xlabel('rank index of items')
plt.ylabel('population')
