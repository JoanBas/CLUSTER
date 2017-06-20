"""
PARTS OF THE PROGRAM:
1. Generate Configuration.
2. Compute Parameters of the nonhomogneus PP.
3. Estimate Cumulative distrivution.
4. Compute Area Test Statistic.
5. Compare with the previous configuration, choose one.
5. Modify Configuration.
7. Back to 2.
"""

import os
import random
import numpy as np
import math
import datetime
from matplotlib import pyplot as plt
from scipy.optimize import minimize
import time
from data import dataset
from mpl_toolkits.mplot3d import Axes3D

xs__=[0]

def del_repeated(ts):
    e_ant = "-"
    for e in range(len(ts)-1,-1,-1):
        if e_ant==ts[e]:
            del ts[e]
        e_ant=ts[e]
    return ts

def f(x):
    return x.total_seconds()
f = np.vectorize(f)


def Modify_Configuration(ai):
    pm=0.5#probability to merge
    ps=0.2 #probability to split
    random.seed(time.time())
    #MERGE:
    ai_=[]
    a_ant=[]
    merge=False
    for a in ai:
        if merge==True:
            ai_.append(a_ant+a)
            merge=False
            continue
        p = random.random()
        if p<pm and a!=ai[-1]:
            merge=True
            a_ant=a
        else:
            ai_.append(a)
    #SPLIT
    ai=ai_
    ai_=[]
    for a in ai:
        p = random.random()
        if p<ps and len(a)!=1:
            n=random.choice(range(1,len(a)))
            ai_.append(a[:n])
            ai_.append(a[n:])
        else:
            ai_.append(a)
    return ai_

def simulate_uni_hawkes(alpha, beta, mu,  n, plot=0):
    arrivals =[]
    eps= 1e-6

    #recurrence function
    def S(k):
        if k==1: return 1
        return 1+S(k-1)*math.exp(-beta*(arrivals[k-1]-arrivals[k-2]))

    #function to be solved to obtain u
    def fu(u,U,k):
        return math.log(U) + mu* (u-arrivals[k-1]) + alpha/beta * S(k)* (1-math.exp(-beta*(u-arrivals[k-1])))


    def fu_prime(u, U, k):
        return mu + alpha * S(k) * math.exp(-beta * (u - arrivals[k-1]))

    #iteration procedure to solve f(u)
    def solve_u(U,k):
        u_prev=arrivals[k-1]-math.log(U)/mu
        u_next=u_prev-fu(u_prev,U,k)/fu_prime(u_prev,U,k)
        while abs(u_next-u_prev)>eps:
            u_prev=u_next
            u_next=u_prev-fu(u_prev,U,k)/fu_prime(u_prev,U,k)
        return 0.5*(u_prev+u_next)

    #FALTA SET SEED
    t1=-math.log(random.random())/mu
    arrivals.append(t1)
    k=len(arrivals)
    while k<n:
        U=random.random()
        t_next=solve_u(U,k)
        arrivals.append(t_next)
        k=len(arrivals)

    arrivals=np.asarray(arrivals)
    if plot:
        ys = [1] * len(arrivals)
        plt.plot(arrivals,ys,"r|")
        plt.show()

    return arrivals

def loglik(params):

    alpha_i=params[0]**2
    beta_i = params[1]**2
    mu_i = params[2]**2

    def funcio(z):
        return np.sum(np.exp(-beta_i * (arrivals[z] - arrivals[:z])))
    funcio = np.vectorize(funcio)

    sum=0
    for arrivals in xs__[0]:
        n=len(arrivals)

        term_1=-mu_i*arrivals[n-1]

        term_2 = np.sum(alpha_i/beta_i*(np.exp( -beta_i * (arrivals[n-1] - arrivals)) - 1))
        Ai=funcio(np.arange(1,n))
        Ai= np.concatenate(([0],Ai))

        term_3 = np.sum(np.log(mu_i + alpha_i * Ai))

        sum+=-term_1 - term_2 - term_3

    return sum

def find_values(x0):
    res = minimize(loglik, x0, method='nelder-mead', options={'xtol': 1e-8, 'disp': True, 'maxiter':1000})
    if res.success:
        params = res.x ** 2
        loglikelihood = np.asarray(loglik(params))
        print params, loglikelihood
        return np.append(res.x**2,loglikelihood)
    else: return [False,False,False]

def train_hawkes_parameters(k):
    x_user = users[k]
    x0=np.random.rand(3)
    print "rands:", x0
    for i in range(len(x_user)-1,-1,-1):
        if len(x_user[i])<5: del x_user[i]
        else:
            inici=x_user[i][0]
            x_user[i]=np.asarray(x_user[i])-inici
    xs__[0]=x_user

    params=find_values(x0)
    print params
    return params

if __name__ == "__main__":
    data=dataset()
    users = data.prepare()

    np.random.seed(int(time.time()))

    params = []
    file=open("Files/in_sess.csv","wt")

    for k in users.keys():
        params= train_hawkes_parameters(k)
        if any(params): file.write(k+","+str(params[0])+","+str(params[1])+","+str(params[2])+","+str(params[3])+"\n")

    file.close()