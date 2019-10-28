import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import pygmo as pg
from sklearn.metrics import mean_absolute_error

###################################################################################################
"""
Initial Recommendation list with C items for each user
"""
Targets=np.genfromtxt('.../TestML.csv',delimiter=',')
Trainings=np.genfromtxt('.../TrainML.csv',delimiter=',')
Predictions=np.genfromtxt('.../PredictionML.csv',delimiter=',')
Rating_Matrix=np.genfromtxt('.../RatingMatrix.csv',delimiter=',')

C=50 #lenght of initial recommendation list for each user

nonzero=[]
for i in range(len(Targets)):
    nonzero.append(np.count_nonzero(Targets[i]))

Predictions_sort=np.argsort(-Predictions, axis=1)
Predictions_sort_value = np.zeros((len(Rating_Matrix), len(Rating_Matrix.T)))
for i in range(len(Predictions)):
    Predictions_sort_value[i] = Predictions[i][Predictions_sort[i]]

Targets_sort_value = np.zeros((len(Rating_Matrix), len(Rating_Matrix.T)))
for i in range(len(Targets)):
    Targets_sort_value[i] = Targets[i][Predictions_sort[i]]

for i in range(len(Predictions_sort)):
    for j in range(len(Predictions_sort.T)):
        if Predictions_sort_value[i][j]==0:
            Predictions_sort[i][j]=-1   #no item to be recommended when reach -1

recommend=np.zeros((len(Rating_Matrix), C))
for i in range(C):
    recommend[:, ][:, i]=Predictions_sort[:, ][:, i]


#Predictions[0][int(recommend[0][0])] #the predictions of those items
###################################################################################################
"""
Unpopularity:
through mean and variance T.Jambor(2010)
"""
mean = np.true_divide(Trainings.sum(0),(Trainings!=0).sum(0))

c2 = Trainings[:]
c2 = c2.astype('float')
c2[c2 == 0]=np.NaN
Var = np.nanvar(c2,axis=0)

UnPop = 1/(mean*(Var+1)**2)
###################################################################################################
"""
Item Provider binary matrix
"""
ItemProvider = pd.read_csv('.../prov_ML1M.csv',delimiter=',',header = None,encoding='latin-1')  # through IMDBpython.py
prov = np.array(ItemProvider)
###################################################################################################
"""
MOEA
"""
###
def selection(pop): # k: lenght of list
    pop1 = list(filter(lambda x: x != -1, pop))  # pop -1 s
    parent1 = random.sample(pop1, k)
    parent2 = random.sample(pop1, k)
    return parent1 ,parent2
###
def fitness(Predictions_row, p1, p2, Targets_row):

    Sigma_pred1 = Predictions_row[np.array(p1).astype(int)].mean()
    Sigma_pred2 = Predictions_row[np.array(p2).astype(int)].mean()
    Sigma_Target1 = Targets_row[np.array(p1).astype(int)].mean()
    Sigma_Target2 = Targets_row[np.array(p2).astype(int)].mean()
    Sigma_unpop1 = UnPop[np.array(p1).astype(int)].mean()
    Sigma_unpop2 = UnPop[np.array(p2).astype(int)].mean()

    providerNum = 8
    pro1 = prov[np.array(p1).astype(int)]
    pro2 = prov[np.array(p2).astype(int)]
    a=[]
    b=[]
    for j in range(len(prov.T)):
        a.append(pro1[0][j] | pro1[1][j] | pro1[2][j] | pro1[3][j] | pro1[4][j])
        b.append(pro2[0][j] | pro2[1][j] | pro2[2][j] | pro2[3][j] | pro2[4][j])
    # print("p_coverage for p1= ",sum(a)/len(prov.T))
    p1_cov = sum(a)
    p2_cov = sum(b)
    if abs(p1_cov - p2_cov) <= 3:
        if (Sigma_pred1 >= Sigma_pred2 and Sigma_unpop1 >= Sigma_unpop2) or (
                Sigma_pred1 > Sigma_pred2 and Sigma_unpop1 < Sigma_unpop2):
            Sigma_pred = Sigma_pred1
            Sigma_Target = Sigma_Target1
            Sigma_unpop = Sigma_unpop1
            sigma_pFair = p1_cov
            p = p1
        elif (Sigma_pred2 >= Sigma_pred1 and Sigma_unpop2 >= Sigma_unpop1) or (
                Sigma_pred1 < Sigma_pred2 and Sigma_unpop1 > Sigma_unpop2):
            Sigma_pred = Sigma_pred2
            Sigma_Target = Sigma_Target2
            Sigma_unpop = Sigma_unpop2
            sigma_pFair = p2_cov
            p = p2
    elif (p1_cov - p2_cov) >= 4:
        Sigma_pred = Sigma_pred1
        Sigma_Target = Sigma_Target1
        Sigma_unpop = Sigma_unpop1
        sigma_pFair = p1_cov
        p = p1
    elif (p2_cov - p1_cov) >= 4:
        Sigma_pred = Sigma_pred2
        Sigma_Target = Sigma_Target2
        Sigma_unpop = Sigma_unpop2
        sigma_pFair = p2_cov
        p = p2


    return Sigma_pred, Sigma_unpop, sigma_pFair, p, Sigma_Target
###
def crossover(p1, p2):
    k=5 # lenght of list
    off1 = []
    off2 = []
    pointer = np.random.randint(1,k)
    off1 = p1[:pointer] + p2[pointer:]
    off2 = p2[:pointer] + p1[pointer:]

    # eliminate duplicates
    while True:
        if len(off1) != len(set(off1)):
            dupes = [x for n, x in enumerate(off1) if x in off1[:n]]  # duplicate elements
            # print(dupes)
            for h in range(len(dupes)):  # the indexes of duplicates
                index = [i for i, x in enumerate(off1) if x == dupes[h]]
                for g in range(len(index)):
                    if off1[index[g]] != p1[index[g]]:
                        off1[index[g]] = p1[index[g]]
        dupes = [x for n, x in enumerate(off1) if x in off1[:n]]
        if dupes == []:
            break

    while True:
        if len(off2) != len(set(off2)):
            dupes = [x for n, x in enumerate(off2) if x in off2[:n]]  # duplicate elements
            # print(dupes)
            for h in range(len(dupes)):  # the indexes of duplicates
                index = [i for i, x in enumerate(off2) if x == dupes[h]]
                for g in range(len(index)):
                    if off2[index[g]] != p2[index[g]]:
                        off2[index[g]] = p2[index[g]]
        dupes = [x for n, x in enumerate(off2) if x in off2[:n]]
        if dupes == []:
            break

    return off1, off2
###
def mutation(pop, off1, off2, p1, p2):
    k = 5  # lenght of list
    pop1 = list(filter(lambda x: x != -1, pop)) # remove -1 s
    pointer = np.random.randint(0, k)
    while True:
        a = random.sample(list(pop1), 1)[0]
        if (a not in p1) and (a not in p2):
            off1[pointer] = a
            break

    pointer = np.random.randint(0, k)
    while True:
        a = random.sample(list(pop1), 1)[0]
        if (a not in p1) and (a not in p2):
            off2[pointer] = a
            break

    return off1, off2
###################################################################################################
"""
main
"""
k=5 # length of list
gens = 30 # number of generations
NP = 80 #size of population
providerNum = 8 # or len(prov.T) : number of providers (movie companies)

Sigma_pred = np.zeros((NP,len(recommend)))
Sigma_Target = np.zeros((NP,len(recommend)))
Sigma_unpop = np.zeros((NP,len(recommend)))
Sigma_pFair = np.zeros((NP,len(recommend)))
lists=[]


for y in range(NP):
    print(y)
    for i in range(len(recommend)):
        #print(i)
        ui = recommend[i]
        if np.count_nonzero(ui + 1) >= 11:  # ui+1 : because of -1 s
            pp = selection(ui)
            p1 = pp[0]
            p2 = pp[1]
            # fitness(Predictions[i], p1, p2)
            f = fitness(Predictions[i], p1, p2, Targets[i])
            Sigma_pred_max = f[0]
            Sigma_unpop_max = f[1]
            Sigma_pFair_max = f[2]
            lists_max = f[3]
            Sigma_Target_max = f[4]

            for j in range(gens):
                #print(j)
                # crossover (two offsprings)
                c = crossover(p1, p2)
                off1 = c[0]
                off2 = c[1]
                # mutation (two offsprings)
                m = mutation(ui, off1, off2, p1, p2)
                off1 = m[0]
                off2 = m[1]
                s = fitness(Predictions[i], off1, off2, Targets[i])
                if (s[2]-Sigma_pFair_max) >= 3:
                    Sigma_pred_max = s[0]
                    Sigma_unpop_max = s[1]
                    Sigma_pFair_max = s[2]
                    lists_max = s[3]
                elif (s[0] >= Sigma_pred_max and s[1] >= Sigma_unpop_max) or (s[0] > Sigma_pred_max and s[1] < Sigma_unpop_max):
                    Sigma_pred_max = s[0]
                    Sigma_unpop_max = s[1]
                    Sigma_pFair_max = s[2]
                    lists_max = s[3]
                # print("Sigma_pred_max: {}".format(Sigma_pred_max), " Sigma_unpop_max: {}".format(Sigma_unpop_max), " lists_max: {}".format(lists_max))
                p1 = off1
                p2 = off2
            Sigma_pred[y][i] = Sigma_pred_max
            Sigma_unpop[y][i] = Sigma_unpop_max
            Sigma_Target[y][i] = Sigma_Target_max
            Sigma_pFair[y][i] = Sigma_pFair_max
            lists.extend(lists_max)

        elif np.count_nonzero(ui + 1) <= k:
            lists_max = ui[np.nonzero(ui+1)]
            Sigma_pred[y][i] = Predictions[i][lists_max.astype(int)].mean()
            Sigma_Target[y][i] = Targets[i][lists_max.astype(int)].mean()
            Sigma_unpop[y][i] = UnPop[lists_max.astype(int)].mean()

            pro = prov[lists_max.astype(int)]
            a = []
            for j in range(len(prov.T)):
                kk=np.count_nonzero(ui + 1)
                if kk==5:
                    a.append(pro[0][j] | pro[1][j] | pro[2][j] | pro[3][j] | pro[4][j])
                elif kk==4:
                    a.append(pro[0][j] | pro[1][j] | pro[2][j] | pro[3][j])
                elif kk==3:
                    a.append(pro[0][j] | pro[1][j] | pro[2][j])
                elif kk==2:
                    a.append(pro[0][j] | pro[1][j])
                elif kk==1:
                    a.append(pro[0][j])
                elif kk==0:
                    a.append(0)
            p_cov = sum(a)
            Sigma_pFair[y][i] = p_cov

            lists.extend(lists_max)
            zeroo = k-len(lists_max)
            lists.extend([-1] * zeroo)

        else:
            lists_max = ui[:k]
            Sigma_pred[y][i] = Predictions[i][lists_max.astype(int)].mean()
            Sigma_Target[y][i] = Targets[i][lists_max.astype(int)].mean()
            Sigma_unpop[y][i] = UnPop[lists_max.astype(int)].mean()

            pro = prov[lists_max.astype(int)]
            a = []
            for j in range(len(prov.T)):
                a.append(pro[0][j] | pro[1][j] | pro[2][j] | pro[3][j] | pro[4][j])
            p_cov = sum(a)
            Sigma_pFair[y][i] = p_cov

            lists.extend(lists_max)

listss=np.array(lists).reshape(NP,len(recommend)*k)

Sigma_pred[np.isnan(Sigma_pred)] = 0
Sigma_unpop[np.isnan(Sigma_unpop)] = 0
Sigma_Target[np.isnan(Sigma_Target)] = 0
Sigma_pFair[np.isnan(Sigma_pFair)] = 0

#show the generated lists for user 0
user_index=0
li = listss[:,user_index*k:(user_index*k)+k]
lis = list(filter(lambda x: x != -1, np.unique(li)))
#print("lists generated by GA for user {} is \n{}: ".format(user_index, li))
#print("and {} unique items are: {}".format(len(lis),lis))

#####################################################################################################
"""
Pareto Front
"""
def identify_pareto(points):
    # Count number of items
    population_size = points.shape[0]
    # Create a NumPy index for scores on the pareto front (zero indexed)
    population_ids = np.arange(population_size)
    # Create a starting list of items on the Pareto front
    # All items start off as being labelled as on the Parteo front
    pareto_front = np.ones(population_size, dtype=bool)
    # Loop through each item. This will then be compared with all other items
    for i in range(population_size):
        # Loop through all other items
        for j in range(population_size):
            # Check if our 'i' pint is dominated by out 'j' point
            if all(points[j] >= points[i]) and any(points[j] > points[i]):
                # j dominates i. Label 'i' point as not on Pareto front
                pareto_front[i] = 0
                # Stop further comparisons with 'i' (no more comparisons needed)
                break
    # Return ids of scenarios on pareto front
    return population_ids[pareto_front]

PF_total=[]
for pf in range(len(Sigma_pred.T)):  #users
    r = Sigma_pred[:,][:,pf]
    un = Sigma_unpop[:,][:,pf]
    fair = Sigma_pFair[:, ][:, pf]

    PF = []
    points = []
    for ii in range(len(r)):  #NPs
        points.append([-r[ii], -un[ii], -fair[ii]])
    if len(np.unique(points))== 2:
        PF.append(np.array([0]))
    else:
        #pareto = identify_pareto(-np.array(points))
        #PF.append(-np.array(points)[pareto])
        PF.append(identify_pareto(-np.array(points)))
    PF_total.append([PF])

# plot PFs for user 0
pf=0

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

n = list(np.arange(1, NP + 1))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
r = Sigma_pred[:, ][:, pf]
un = Sigma_unpop[:, ][:, pf]
fair = Sigma_pFair[:, ][:, pf]
print("PFs for user {} is \n{}: ".format(pf, PF_total[pf][0][0]))

ax.scatter(r, un, fair, c='r', marker='o')
for i, txt in enumerate(n):
    ax.annotate(txt, (r[i], un[i]))

ax.scatter(r[PF_total[pf][0][0]], un[PF_total[pf][0][0]], fair[PF_total[pf][0][0]], label='PF', s=100, marker=(5, 1))
#ax.plot(r[PF_total[pf][0][0]], un[PF_total[pf][0][0]], fair[PF_total[pf][0][0]], color='r')
ax.set_xlabel('prediction average')
ax.set_ylabel('unpopularity average')
ax.set_zlabel('provider coverage')
plt.legend()
plt.show()
#####################################################################################################

"""
evaluation for CF model : accuracy
"""
CF=recommend[:,:k]
mae = []
for i in range(len(recommend)):
    mae.append(mean_absolute_error(Predictions[i][CF[i].astype(int)], Targets[i][CF[i].astype(int)]))
MAE_CF = np.array(mae).mean()
print("MAE for CF (1 top-k list for each user wrt accuracy only): ",MAE_CF)

#####################################################################################################
"""
evaluation for MOEA model :accuracy
"""
#show the generated lists for user 0
user_index=0
li = listss[:,user_index*k:(user_index*k)+k]
lis = list(filter(lambda x: x != -1, np.unique(li)))
print("PF lists generated for user {} is \n{}: ".format(user_index, li[PF_total[user_index][0][0]]))

mae1 = []
for i in range(len(Predictions)):
    user_index = i
    li = listss[:, user_index * k:(user_index * k) + k]
    a = Predictions[i][li[PF_total[i][0][0]].astype(int)]
    b = Targets[i][li[PF_total[i][0][0]].astype(int)]
    mae1.append(mean_absolute_error(a,b))
MAE_MOEA_avg = np.array(mae1).mean()
print("MAE for MOEA lists (average): ",MAE_MOEA_avg)

mae_min = []
for i in range(len(Predictions)):
    user_index = i
    li = listss[:, user_index * k:(user_index * k) + k]
    a = Predictions[i][li[PF_total[i][0][0]].astype(int)]
    b = Targets[i][li[PF_total[i][0][0]].astype(int)]
    mae_eachuser = []
    for j in range(len(PF_total[user_index][0][0])):
        mae_eachuser.append(mean_absolute_error(a[j], b[j]))
    mae_min.append(min(mae_eachuser))
MAE_MOEA_min = np.array(mae_min).mean()
print("MAE for MOEA lists (Minimum): ",MAE_MOEA_min)

mae_max = []
for i in range(len(Predictions)):
    user_index = i
    li = listss[:, user_index * k:(user_index * k) + k]
    a = Predictions[i][li[PF_total[i][0][0]].astype(int)]
    b = Targets[i][li[PF_total[i][0][0]].astype(int)]
    mae_eachuser = []
    for j in range(len(PF_total[user_index][0][0])):
        mae_eachuser.append(mean_absolute_error(a[j], b[j]))
    mae_max.append(max(mae_eachuser))
MAE_MOEA_max = np.array(mae_max).mean()
print("MAE for MOEA lists (Maximum): ",MAE_MOEA_max)

#####################################################################################################
"""
evaluation for PF provider coverage
"""
providerNum=8

p_covv=[]
for i in range(len(PF_total)):
    li = listss[:, i * k:(i * k) + k]
    #lis = np.array(list(filter(lambda x: x != -1, np.unique(li))))
    ll = []
    for j in range(len(PF_total[i][0][0])):
        ll=list(li[PF_total[i][0][0]][0])
        ll.extend(list(li[PF_total[i][0][0]][j]))
    lll=list(filter(lambda x: x != -1, np.unique(ll)))
    proo = prov[np.array(lll).astype(int)]
    proo_df = pd.DataFrame(proo, columns=('p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8'))
    p_covv.append(np.count_nonzero(proo_df.sum(axis=0)))

print("average of PF provider coverage for all users", (np.array(p_covv)/providerNum *100).mean())
#####################################################################################################
"""
evaluation for CF provider coverage
"""

p_covv_cf=[]
for i in range(len(PF_total)):
    cf=recommend[i][:k]
    lll=list(filter(lambda x: x != -1, np.unique(cf)))
    proo = prov[np.array(lll).astype(int)]
    proo_df = pd.DataFrame(proo, columns=('p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8'))
    p_covv_cf.append(np.count_nonzero(proo_df.sum(axis=0)))

print("average of CF provider coverage for all users", (np.array(p_covv_cf)/providerNum *100).mean())
#####################################################################################################
"""
evaluation for CF long tail coverage
"""
per=0.7
LT_cov_CF=[]
for i in range(len(PF_total)):
    cf = recommend[i][:k]
    lll = list(filter(lambda x: x != -1, np.unique(cf)))
    unpop = UnPop[np.array(lll).astype(int)]
    a = [i for i, x in enumerate(unpop >= np.quantile(UnPop, per)) if x]
    LT_cov_CF.extend(np.array(lll)[a])

count=0
for i in range(len(UnPop)):
    if UnPop[i]>=np.quantile(UnPop,per):
        count+=1

print("average of CF Long Tail coverage for all users", len(np.unique(np.array(LT_cov_CF)))/count)

#####################################################################################################
"""
evaluation for PF long tail coverage
"""
LT_cov=[]
for i in range(len(PF_total)):
    li = listss[:, i * k:(i * k) + k]
    ll = []
    for j in range(len(PF_total[i][0][0])):
        ll=list(li[PF_total[i][0][0]][0])
        ll.extend(list(li[PF_total[i][0][0]][j]))
    lll=list(filter(lambda x: x != -1, np.unique(ll)))
    unpop = UnPop[np.array(lll).astype(int)]
    a = [i for i, x in enumerate(unpop >= np.quantile(UnPop, per)) if x]
    LT_cov.extend(np.array(lll)[a])

count=0
for i in range(len(UnPop)):
    if UnPop[i]>=np.quantile(UnPop,per):
        count+=1

print("average of PF Long Tail coverage for all users", len(np.unique(np.array(LT_cov)))/count)

