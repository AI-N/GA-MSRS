
from __future__ import print_function, division
from builtins import range, input

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from math import sqrt

from keras.models import Model
from keras.layers import Input, Embedding, Dot, Add, Flatten, Dense, Concatenate
from keras.layers import Dropout, BatchNormalization, Activation
from keras.regularizers import l2
from keras.optimizers import SGD, Adam

# load in the data
df = pd.read_csv('.../ml-1m/edited_rating.csv')
# edited_rating matrix: mapping unsequential movie ids to sequential starts with 0
# userId 	movieId	  rating   movie_idx
# 2598	      1	        4	      0
# 17	      1	        4	      0
# 4088	      1	        5	      0


N = df.userId.max() + 1 # number of users
M = df.movie_idx.max() + 1 # number of movies

# split into train and test (80-20)
df = shuffle(df)
cutoff = int(0.8*len(df))
df_train = df.iloc[:cutoff]
df_test = df.iloc[cutoff:]

# initialize variables
K = 10 # latent dimensionality
mu = df_train.rating.mean()
epochs = 15
reg = 0.05 # regularization penalty


# keras model
u = Input(shape=(1,))
m = Input(shape=(1,))
u_embedding = Embedding(N, K)(u) # (N, 1, K)
m_embedding = Embedding(M, K)(m) # (N, 1, K)


##### main branch
u_bias = Embedding(N, 1)(u) # (N, 1, 1)
m_bias = Embedding(M, 1)(m) # (N, 1, 1)
x = Dot(axes=2)([u_embedding, m_embedding]) # (N, 1, 1)
x = Add()([x, u_bias, m_bias])
x = Flatten()(x) # (N, 1)


##### side branch
u_embedding = Flatten()(u_embedding) # (N, K)
m_embedding = Flatten()(m_embedding) # (N, K)
y = Concatenate()([u_embedding, m_embedding]) # (N, 2K)
y = Dense(400)(y)
y = Activation('elu')(y)
# y = Dropout(0.5)(y)
y = Dense(1)(y)


##### merge
x = Add()([x, y])

model = Model(inputs=[u, m], outputs=x)
model.compile(
  loss='mse',
  # optimizer='adam',
  # optimizer=Adam(lr=0.01),
  optimizer=SGD(lr=0.08, momentum=0.9),
  metrics=['mse'],
)

r = model.fit(
  x=[df_train.userId.values, df_train.movie_idx.values],
  y=df_train.rating.values - mu,
  epochs=epochs,
  batch_size=128,
  validation_data=(
    [df_test.userId.values, df_test.movie_idx.values],
    df_test.rating.values - mu
  )
)


# plot losses
plt.plot(r.history['loss'], label="train loss")
plt.plot(r.history['val_loss'], label="test loss")
plt.legend()
plt.show()

# plot mse
plt.plot(r.history['mean_squared_error'], label="train mse")
plt.plot(r.history['val_mean_squared_error'], label="test mse")
plt.legend()
plt.show()

#######################################################
# Test Matrix
df_test_array=np.array(df_test)
TestML=np.zeros((N,M))    #target ratings
for i in range(len(df_test_array)):
   TestML[int(df_test_array[i][0])][int(df_test_array[i][3])]=int(df_test_array[i][2])

# Train Matrix
df_train_array=np.array(df_train)
TrainML=np.zeros((N,M))    #train ratings
for i in range(len(df_train_array)):
   TrainML[int(df_train_array[i][0])][int(df_train_array[i][3])]=int(df_train_array[i][2])

# Prediction Matrix
pp = model.predict(x=[df_test.userId.values, df_test.movie_idx.values],batch_size=128)

rr=[]   #predicted ratings
for i in range(len(df_test)):
    rr.append(pp[i]+mu)

PredictionML=np.zeros((N,M))
for i in range(len(df_test_array)):
    PredictionML[int(df_test_array[i][0])][int(df_test_array[i][3])]=rr[i]
    
# Rating Matrix
RatingMatrix=np.zeros((N,M),float)
Init_Matrix=np.array(df)
i=iter(Init_Matrix)
for j in range(len(Init_Matrix)):
    n=next(i)
    x=int(n[0])
    y=int(n[3])
    RatingMatrix[x][y]=n[2]
#############################################################
# calculate accuracy
def mse(p, t):
  p = np.array(p)
  t = np.array(t)
  return np.mean((p - t)**2)
def mae(p, t):
  p = np.array(p)
  t = np.array(t)
  return np.mean(abs(p - t))
def rmse(p, t):
  p = np.array(p)
  t = np.array(t)
  return sqrt(np.mean((p - t)**2))

print('test mse:', mse(PredictionML[np.nonzero(abs(PredictionML))], TestML[np.nonzero(abs(TestML))]))
print('test mae:', mae(PredictionML[np.nonzero(abs(PredictionML))], TestML[np.nonzero(abs(TestML))]))
print('test rmse:', rmse(PredictionML[np.nonzero(abs(PredictionML))], TestML[np.nonzero(abs(TestML))]))

