
from imdb import IMDb
import numpy as np
import pandas as pd
import csv

#extracting title of movies
Movies = pd.read_csv('.../u.item',delimiter='|',header = None,encoding='latin-1')
c1 = []
c2 = []
with open('.../u.item', 'r') as f:
    reader = csv.reader(f, delimiter='|')
    for row in reader:
        c1.append(row[0])
        c2.append(row[1])
mID=c1[:]
mTitle=c2[:]

#print('Movies info:\n',Movies.head())
a=Movies.ix[np.arange(1682),[0,1]]
Movies_Info=np.array(a)
Init_Movies=pd.DataFrame(data=Movies_Info,index=range(len(Movies_Info)),columns=['mID','mTitle'])

# create an instance of the IMDb class
ia = IMDb()
cc=[0, 0]
c=np.array(cc)
movies = mTitle
movie_comp = []

for i in movies:
    print(i)
    moviee = ia.search_movie(i)
    if moviee == []:
        movie_comp.append(c)
    else:
        movie = moviee[0]
        ia.update(movie)
        movie_company = movie.get('production companies')
        movie_com = np.array(movie_company)
        movie_comp.append(movie_com)

bb=np.array(movie_comp[0])
print("Movie company for first movie: ",bb)

movie_comp_id = np.zeros((len(movie_comp), 40)) # 40 is chosen to cover max number of providers for each movie
movie_comp_name = []
for i in range(len(movie_comp)):
    print(i)
    if str(movie_comp[i]) == 'None':
        for k in range(40):
            movie_comp_name.append('N')
    elif str(movie_comp[i]) != 'None':
        mc = np.array(movie_comp[i])
        if mc[0] == 0:
            # movie_comp_id[i][j] = 0
            for k in range(40):
                movie_comp_name.append('N')
        else:
            for j in range(len(movie_comp[i])):
                movie_comp_id[i][j] = mc[j].companyID
                movie_comp_name.append(mc[j].get('name'))

        if mc[0]!=0:
            for k in range(40 - len(movie_comp[i])):
                movie_comp_name.append('N')

movie_comp_namee = np.array(movie_comp_name)
m_comp_name=movie_comp_namee.reshape(1682,40)

# build the ItemProvider matrix with items in rows and providers in columns
# Note that in this work we just consider the major movie companies according to https://en.wikipedia.org/wiki/Major_film_studio
