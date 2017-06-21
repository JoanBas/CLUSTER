
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
from plots import plot_timeline_with_intensity,plot_timelines_multivariate
import csv


def simulate_multi_hawkes(alpha, beta, mu, N_or_T_max, reference_stop_event="none", maxN_total=1000, plot=0, time=0):
    M=len(mu)
    events=[[] for j in range(M)]
    N=np.zeros([M])
    s = 0

    def landa(m,t):
        l=mu[m]
        for n in range(M):
            for t_ in events[n]:
                l+=alpha[m,n]*np.exp(-beta[m,n]*(t-t_))
        return l

    def I(k, t):
        I = 0
        for m in range(k+1):
            I+=landa(m,t)
        return I

    def attribution(s, I_,D):
        for m in range(M):
            if D <= I(m, s) / I_: return m

    end=False
    while not end:
        I_max=I(M-1,s)

        U=random.random()
        w= -np.log(U)/I_max
        s+=w
        D=random.random()

        if D<= I(M-1,s) / I_max:
            #here we choose the criteria to stop simulating
            if time: N_or_T_ref=s
            elif reference_stop_event=="none": N_or_T_ref=np.sum(N)
            else: N_or_T_ref= N[reference_stop_event]

            if N_or_T_ref<N_or_T_max and np.sum(N)<maxN_total:
                k=attribution(s,I_max,D)
                N[k]+=1
                events[k].append(s)
            else: end=True

    for i in range(len(events)):
        events[i]=np.asarray(events[i])
    return events




xs=[]
iexterna=[0]
def loglik(params):
    M = 2
    alpha = np.reshape(params[0:M ** 2], [M, M]) ** 2
    beta = np.reshape(params[M ** 2:2 * M ** 2], [M, M]) ** 2
    mu = np.reshape(params[2 * M ** 2:], [M]) ** 2
    #if iexterna[0]%10000==0:print mu, "\n", alpha, "\n", beta, "\n"
    iexterna[0]+=1

    def R_(m, n, l):
        if l==-1: return 0.

        tl = sequence[m][l]
        if l == 0:
            tl_ant = 0.
        else:
            tl_ant = sequence[m][l - 1]

        if m == n:
            if l != 0: val = np.exp(-beta[m, n] * (tl - tl_ant)) * (1 + R_(m, n, l - 1))
            else: val=0
        else:
            sum = 0
            for tn in sequence[n]:
                if tn < tl and tn >= tl_ant:
                    sum += np.exp(-beta[m, n] * (tl - tn))
            val = np.exp(-beta[m, n] * (tl - tl_ant)) * R_(m, n, l - 1) + sum

        Rmnl[n].append(alpha[m,n]*val)
        return val

    result = 0
    for sequence in xs:
        T=sequence[-1]
        for m in range(M):
            term_1 = -mu[m] * T
            #print "Term1:", term_1

            term_2 = 0
            for n in range(M):
                pre_term_2 = 0
                for event in sequence[n]:
                    pre_term_2 += 1 - np.exp(-beta[m, n] * (T - event))
                pre_term_2 *= alpha[m, n] / beta[m, n]
                term_2 -= pre_term_2
            #print "Term2:", term_2

            term_3 = 0

            Rmnl = [[] for n in range(M)]
            for n in range(M):
                R_(m, n, len(sequence[m]) - 1)
            #print Rmnl

            for l in range(len(sequence[m])):
                pre_term_3 = mu[m]
                for n in range(M):
                    pre_term_3 += Rmnl[n][l]
                term_3 += np.log(pre_term_3)
            #print "Term3:", term_3
            result += term_1 + term_2 + term_3
            #print term_1 + term_2 + term_3
    #regularization=np.sum(alpha)+np.sum(1/beta)
    #print result
    return -result#+regularization


###################################################


def find_values(x0):
    #bounds=((0,2),(0,2),(0,2),(0,2),(0.1,1),(0.1,1),(0.1,1),(0.1,1),(None,None),(None,None))
    #res = minimize(loglik, x0, method='L-BFGS-B',bounds=bounds, options={'disp': False, 'maxiter':10000})
    res = minimize(loglik, x0, method='nelder-mead', options={'xtol': 1e-3, 'disp': False, 'maxiter':10000})
    if res.success:
        params = res.x ** 2
        loglikelihood = np.asarray(loglik(params))
        return np.append(params,loglikelihood)
    else: return [False,False,False]

def train_hawkes_parameters(events_list):
    del xs[:]
    print "number of sessions:  ", len(events_list)
    x0=np.random.rand(10)*2
    for i in range(len(events_list)):#range(len(events_list)-1,-1,-1):
        if len(events_list[i][0]) + len(events_list[i][1])<1: continue #AIXO CAL TREUREHO QUAN HI HAGI LES BONES SESSIOOOONS
        inici=events_list[i][-1][0]
        T=events_list[i][-1][1] - events_list[i][-1][0]

        xs.append( [np.asarray(events_list[i][0]) - inici, np.asarray(events_list[i][1]) - inici, T] )
    params=find_values(x0)
    print params
    return params


if __name__ == "__main__":

    M = 2

    event_kinds = ['views', 'edits']
    dataset_name = "handson3"
    plot_and_compare=False


    #Open and prepare database
    data = dataset(dataset_name = dataset_name, event_kinds=event_kinds)
    users = data.prepare_multivariate(add_start_end_sess=1)

    event_kinds="_".join(str(x) for x in event_kinds)

    #read the already calculated parameters
    try:

        reader = csv.reader(open('Files/in_sess_multi_' + dataset_name + '_' +event_kinds+'.csv', 'r'))
        parameters = {}
        for row in reader:
            parameters[row[0]] = np.asarray(row[1:]).astype(np.float32)
    except: parameters={}
    Nupdates=0

    ntrained=0
    for k in data.IDs:
        print '\n\n\nTrained ', ntrained, " of ", len(data.IDs)
        print '\n\nTraining the parameters for the user ', k
        ntrained+=1
        print users[k]
        params= train_hawkes_parameters(users[k])

        if not any(params): continue

        if plot_and_compare:
            alpha = np.reshape(params[0:M ** 2], [M, M])
            beta = np.reshape(params[M ** 2:2 * M ** 2], [M, M])
            mu = np.reshape(params[2 * M ** 2:2 * M ** 2 + M], [M])

            for i in range(3):
                x_reals = [random.choice(users[k]) for i in range(10)]

                T = float("-inf")
                N = 0
                for x_real in x_reals:
                    N_ = 0
                    for sec in x_real:
                        N_ += len(sec)
                        try:
                            val = sec[-1] - sec[0] + 1
                        except:
                            val = float("-inf")
                        T = max(T, val)
                    N = max(N, N_)
                print T
                x = simulate_multi_hawkes(alpha, beta, mu, T, time=1)
                print len(x[0]), len(x[1])
                plot_timeline_with_intensity(x, alpha, beta, mu, T)
                plot_timelines_multivariate([x] + x_reals)

        if k in parameters:
            print "old: ", parameters[k][-1], " new:", params[-1]
            if params[-1]>parameters[k][-1]: continue

        parameters[k] = params
        print k, " UPDATED!"

        Nupdates+=1




    print "Number of updates: ", Nupdates
    #update parameters file
    filename='Files/in_sess_multi_' + dataset_name + '_' + event_kinds + '.csv'
    print filename, "has been saved"
    file = open(filename, "wt")
    print len(parameters.keys())
    for k in parameters.keys():
        params=parameters[k]
        if any(params):
            s = k
            for param in params:
                s += "," + str(param)
            file.write(s + "\n")
    file.close()

    data = np.genfromtxt('Files/in_sess_multi_' + dataset_name +'_'+ event_kinds+'.csv', delimiter=',')

    data_dict={}
    for l in data:
        data_dict[str(int(l[0]))]=l[1:4]





















"""
#likelihood CHECK


xs=[[[1,3],[2,4]]]
alpha=np.asarray([[0.2,0.1],[0.5,0.1]])
beta=np.asarray([[0.8,0.6],[1.,0.7]])
mu=np.asarray([0.1,0.5])

print np.asarray(loglik(np.append(np.append(alpha,beta),mu).flatten()**(0.5)))




####################################################
alpha=np.asarray([[0.,0.],[0.3,0.1]])
beta=np.asarray([[1.,1.],[1.,1.]])
mu=np.asarray([0.3,0.])


alpha=np.asarray([[0.5]])
beta=np.asarray([[1.]])
mu=np.asarray([0.5])


x=simulate_multi_hawkes(alpha,beta,mu,30)
plot_timeline_with_intensity(x,alpha,beta,mu,30)
exit()

for i in range(3):
    x=simulate_multi_hawkes(alpha,beta,mu,100)
    xs.append(x)
    plot_timeline_with_intensity(x, alpha, beta, mu, 100)
    print xs[i]

avg_params=np.empty([3])
for i in range(10):

    x0 = np.random.rand(10)

    x0=np.append(np.append(alpha, beta), mu).flatten()

    res = minimize(loglik, x0, method='Powell', options={'xtol': 1e-3, 'disp': True, 'maxiter':10000})

    params = res.x ** 2
    #avg_params=np.vstack((avg_params,params))
    loglikelihood = np.asarray(loglik(params**(0.5)))
    real_loglikelihood = np.asarray(loglik(np.append(np.append(alpha,beta),mu).flatten()**(0.5)))

    print params, loglikelihood, real_loglikelihood

    M = 2

    alpha = np.reshape(params[0:M ** 2], [M, M])
    beta = np.reshape(params[M ** 2:2 * M ** 2], [M, M])
    mu = np.reshape(params[2 * M ** 2:], [M])
    x = simulate_multi_hawkes(alpha, beta, mu, 100)
    plot_timeline_with_intensity(x, alpha, beta, mu, 100)
    plot_timelines_multivariate([x] + xs)

alpha=np.asarray([[0.5,0.3],[0.25,0.75]])
beta=np.asarray([[0.9,0.75],[0.5,1.]])
mu=np.asarray([0.5,0.5])
simulate_multi_hawkes(alpha,beta,mu,300)
exit()




def loglik2(params):

    alpha_i=params[0]**2
    beta_i = params[1]**2
    mu_i = params[2]**2

    def funcio(z):
        return np.sum(np.exp(-beta_i * (arrivals[z] - arrivals[:z])))
    funcio = np.vectorize(funcio)

    sum=0
    for arrivals in xs[0]:
        n=len(arrivals)

        term_1=-mu_i*arrivals[n-1]

        term_2 = np.sum(alpha_i/beta_i*(np.exp( -beta_i * (arrivals[n-1] - arrivals)) - 1))
        Ai=funcio(np.arange(1,n))
        Ai= np.concatenate(([0],Ai))

        term_3 = np.sum(np.log(mu_i + alpha_i * Ai))

        sum+=-term_1 - term_2 - term_3

    return sum
def loglik_ant(params):

    M=1

    alpha= np.reshape(params[0:M**2],[M,M])**2
    beta = np.reshape( params[M**2:2*M**2],[M,M])**2
    mu= np.reshape(params[2*M**2:],[M])**2

    def R(m,n,l):
        if l==0: return 0

        if m==n:
            return np.exp(-beta[m,n]*(sequence[m][l]-sequence[m][l-1]))*(1+R(m,n,l-1))
        else:
            sum=0
            for tn in sequence[n]:
                if tn<sequence[m][l] and tn>sequence[m][l-1]:
                    sum+=np.exp(-beta[m,n]*(sequence[m][l]-tn))

            return np.exp(-beta[m,n]*(sequence[m][l]-sequence[m][l-1]))*R(m,n,l-1) + sum

    sequence=xs
    result = 0
    T = max([sec[-1] for sec in sequence])
    for m in range(M):
        event_sequence = sequence[m]

        term_1 = -mu[m]*T
        # print "Term1:", term_1

        term_2 = 0

        for n in range(M):
            pre_term_2=0
            for event in sequence[n]:
                pre_term_2 += 1 - np.exp(-beta[m, n] * (T - event))
            pre_term_2 *= alpha[m, n] / beta[m, n]
            term_2-=pre_term_2
        # print "Term2:", term_2

        term_3 = 0
        for l in range(len(event_sequence)):
            pre_term_3 = mu[m]
            for n in range(M):
                pre_term_3 += alpha[m, n] * R(m, n, l)
            term_3 += np.log(pre_term_3)
        # print "Term3:", term_3
        result -= term_1 + term_2 + term_3

    return result

def simulate_multi_hawkes_ant(alpha, beta, mu, T, plot=0):
    M=len(mu)
    events=[[] for j in range(M)]

    def landa(k,t):
        l=mu[k]
        for n in range(M):
            for t_ in events[n]:
                l+=alpha[k,n]*np.exp(-beta[k,n]*(t-t_))
        return l

    def I(k, t):
        I = 0
        for n in range(k):
            I+=landa(n,t)
        return I

    def attribution(s, I_,D):
        for n in range(M):
            if D <= I(n+1, s) / I_: return n


    #initialization
    i=1
    i_=np.ones([M])
    I_=I(M,0)
    end=False

    #First Event
    U=random.random()
    s=-np.log(U)/I(M,0)
    if s>T: end=True
    dim=attribution(s,I_,random.random())
    last_t=s
    events[dim].append(s)

    #General routine
    while not end:
        step_added=False
        i+=1
        i_[dim]=i_[dim]+1

        #Update Maximum Intensity
        I_=I(M,last_t) # + np.sum(alpha[:,dim]) ALERTA! AQUI NO SE SI HE DE SUMAR AQUESTA SEGONA PART O NO!!!!!!!!!!!!!!!!! EN CAS QUE ESTIGUI MALAMENT, SHA DE CANVIAR TBE EL PRIMER IF DATTIBUTIONS
        while not step_added:
            #New Event
            U=random.random()
            s=s-np.log(U)/I_

            if s>T:
                end=True
                break

            #Attribution-Rejection Test
            D=random.random()
            I_M=I(M, s)
            if D <= I_M  / I_:
                dim=attribution(s,I_,D)
                last_t=s
                events[dim].append(s)
                step_added=True
                print s
            else: I_=I_M

    return events"""

"""if __name__ == "__main__":

    data = dataset()
    users = data.prepare_multivariate()
    M=2

    plot_and_compare=False
    file = open("in_sess_multi.csv", "wt")
    parameters=[]

    for k in data.IDs:
        #print users['1399']
        params= train_hawkes_parameters(k)
        parameters.append(params)

        if any(params):
            s= k
            for param in params:
                s+= ","+str(param)

            file.write(s + "\n")

        if plot_and_compare:
            alpha = np.reshape(params[0:M ** 2], [M, M])
            beta = np.reshape(params[M ** 2:2 * M ** 2], [M, M])
            mu = np.reshape(params[2 * M ** 2:2 * M ** 2+M], [M])

            for i in range(3):
                x_reals= [random.choice(users[k]) for i in range(10)]

                T=float("-inf")
                N=0
                for x_real in x_reals:
                    N_=0
                    for sec in x_real:
                        N_+=len(sec)
                        try:
                            val = sec[-1]-sec[0]+1
                        except:
                            val = float("-inf")
                        T = max(T, val)
                    N=max(N,N_)
                print T
                x = simulate_multi_hawkes(alpha, beta, mu, N)
                print len(x[0]), len(x[1])
                plot_timeline_with_intensity(x, alpha, beta, mu, T)
                plot_timelines_multivariate([x] +x_reals)
    file.close()

    data = np.genfromtxt('in_sess_multi.csv', delimiter=',')

    data_dict={}
    for l in data:
        data_dict[str(int(l[0]))]=l[1:4]
"""
