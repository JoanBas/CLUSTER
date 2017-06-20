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

dataset_name='handson2'
data=dataset()
hour_d=data.average_hour_distribution(data.data['views'],plot=False) #daily average of different users
day_d=data.day_distribution(data.data['views'],plot=False) #weekly average of different users

distributions={}
plot=1
plot_total=1

prep=data.prepare()
sess_len_dist={}
N_weeks=data.timedifference.days/7. #number of weeks
N_acc=[]
sess_per_week_acc=[]
for id in hour_d.keys():
    N_sess=[]
    sessions=prep[id]
    for sess in sessions:
        N_sess.append(len(sess))
    N_acc+=N_sess
    sess_len_dist[id]=N_sess
    sess_per_week=len(sessions)/N_weeks
    sess_per_week_acc.append(sess_per_week)
    distribution = np.empty([0])
    for d in range(7):
        h = hour_d[id] * day_d[id][d]
        distribution = np.append(distribution, h)
    distribution /= np.sum(distribution)
    #distribution*=sess_per_week
    distributions[id] = distribution

    if plot:
        plt.hist(N_sess,bins=max(N_sess)-1)
        plt.show()
        plt.plot(range(24 * 7), distribution)
        print "sessions per week:", sess_per_week
        plt.xticks(np.arange(7) * 24, ["Mon", "Tue", "Wed", "Thur", "Fri", "Sat", "Sun"])
        plt.grid()
        plt.show()


file=open("Files/sess_dist.csv","wt")
for id in distributions.keys():
    string=id
    for val in distributions[id]:
        string+= ","+str(val)
    file.write(string+"\n")
file.close()
print "sess_dist have been saved"

sess_len_dist_combined=[]
file=open("Files/sess_len_dist.csv","wt")
for id in sess_len_dist.keys():
    string=id
    for val in sess_len_dist[id]:
        string+= ","+str(val)
        sess_len_dist_combined.append(val)
    file.write(string+"\n")
file.close()
print "sess_len_dist have been saved"

file=open("Files/sess_len_dist.csv","wt")
string=dataset_name
for val in sess_len_dist[id]:
    string += "," + str(val)
    sess_len_dist_combined.append(val)
file.write(string + "\n")
print "sess_len_dist_combined have been saved"



if plot_total:
    plt.hist(N_acc, bins=max(N_acc) - 1, cumulative=1)
    #plt.yscale('log')
    plt.show()
    plt.hist(sess_per_week_acc)
    plt.show()

exit()


"""distributions={}
plot=0
for id in hour_d.keys():
    distribution=np.empty([0])
    for d in range(7):
        h=hour_d[id]*day_d[id][d]
        distribution=np.append(distribution,h)
    distribution/=np.linalg.norm(distribution)
    if plot:
        plt.plot(range(24*7),distribution)
        plt.xticks(np.arange(7)*24,["Mon", "Tue", "Wed", "Thur","Fri", "Sat", "Sun"])
        plt.grid()
        plt.show()
    distributions[id]=distribution




file=open("sess_dist.csv","wt")
for id in distributions.keys():
    string=id
    for val in distributions[id]:
        string+= ","+str(val)
    file.write(string+"\n")
file.close()
print "sess_dist have been saved"


plot=0
plot_total=0
prep=data.prepare()
sess_len_dist={}
sess_per_week={}
N_weeks=data.timedifference.days/7.
N_acc=[]
for id in prep.keys():
    N_sess=[]
    sessions=prep[id]
    for sess in sessions:
        N_sess.append(len(sess))
    #h=np.histogram(N_sess,bins=range(max(N_sess)+2))
    N_acc+=N_sess
    if plot:
        plt.hist(N_sess,bins=max(N_sess)-1)
        plt.show()
    sess_len_dist[id]=N_sess
    sess_per_week[id]=len(sessions)/N_weeks

if plot_total:
    plt.hist(N_acc, bins=max(N_acc) - 1)
    #plt.yscale('log')
    plt.show()

file=open("sess_len_dist.csv","wt")
for id in sess_len_dist.keys():
    string=id
    for val in sess_len_dist[id]:
        string+= ","+str(val)
    file.write(string+"\n")
file.close()
print "sess_len_dist have been saved"

file=open("sess_per_week.csv","wt")
for id in sess_per_week.keys():
    string=id+","+str(sess_per_week[id])
    file.write(string+"\n")
file.close()
print "sess_per_week have been saved" """