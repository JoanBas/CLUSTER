import inside_session as ins
import numpy as np
import time
import data as dta
import random
import csv
from inside_session import simulate_uni_hawkes
from plots import plot_timelines, cdf
from matplotlib import pyplot as plt
from data import dataset
from datetime import datetime
from inside_session_multi import simulate_multi_hawkes
from sklearn import mixture
import inside_session_multi as ism


def simulate_sessions(parameters,n_sessions,time_per_session):
    sessions=[]
    for i in range(n_sessions):
        sess=ism.simulate_multi_hawkes(parameters[0], parameters[1], parameters[2], N_or_T_max=time_per_session,time=True)
        totalsess=np.append(sess[0],sess[1])
        sess.append([np.min(totalsess), np.max(totalsess)])
        sessions.append(sess)
    return sessions


def find_error(filename="errors.csv",plot=False):
    alpha = np.asarray([[0.3, 1.2], [0.2, 0.1]])
    beta = np.asarray([[0.8, 2.6], [0.4, 0.2]])
    mu = np.asarray([0.2, 0.006])

    parameters=[alpha,beta,mu]


    session=simulate_sessions(parameters=parameters,n_sessions=10, time_per_session=10.)
    print session
    gammas=[]
    for i in range(10):
        gamma = len(session[i][0])/session[i][2][-1]
        gammas.append(gamma)
    gammas=np.asarray(gammas)
    print np.mean(gammas)
    result = ism.train_hawkes_parameters(session)
    print result



find_error()

