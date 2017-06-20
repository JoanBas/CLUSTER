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

def make_session(usr_id,t):
    n=random.choice(sess_len_dist[usr_id])
    alpha, beta, mu = in_sess[usr_id]
    session = simulate_uni_hawkes(alpha, beta, mu, n)
    session+=t
    return session, session[-1]


reader = csv.reader(open('Files/in_sess.csv', 'r'))
in_sess = {}
for row in reader:
    in_sess[row[0]] = np.asarray(row[1:4]).astype(np.float32)

reader = csv.reader(open('Files/sess_dist.csv', 'r'))
sess_dist = {}
for row in reader:
    sess_dist[row[0]] = np.asarray(row[1:]).astype(np.float32)

reader = csv.reader(open('Files/sess_len_dist.csv', 'r'))
sess_len_dist = {}
for row in reader:
    sess_len_dist[row[0]] =  np.asarray(row[1:]).astype(int)

P = dataset()
users = P.users_visits
maxdate = datetime.strptime('9-11-2014 23:59:59', '%d-%m-%Y %H:%M:%S')
mindate = datetime.strptime('3/11/2014 00:00:00', '%d/%m/%Y %H:%M:%S')
users_timelines={}
plotAll=0
for usr_id in in_sess.keys():
    t0=0.
    ti=0.
    t=0.
    timeline=np.empty([1,0])
    for hour_val in sess_dist[usr_id]:
        if t<=ti:
            rand = random.random()
            t+=random.random()*60
        elif t>ti+60:
            rand=1
        elif t>ti and t<ti+60:
            rand=random.random()*(ti+60-t)/60
            t += random.random() * (ti+60-t)
        else: "ALGO PASAAA"
        if hour_val>rand:
            sess, t = make_session(usr_id,t)
            timeline=np.append(timeline,sess)
        ti+=60.
        if t<ti: t=ti
    users_timelines[usr_id]=timeline

    if plotAll:
        plt.figure(1)
        plt.subplot(311)
        plt.plot(range(24 * 7), sess_dist[usr_id])
        plt.xticks(np.arange(7) * 24, ["Mon", "Tue", "Wed", "Thur", "Fri", "Sat", "Sun"])
        plt.axis([0, 24*7, 0, 0.5])
        plt.grid()

        plt.subplot(312)
        usr=users[usr_id]
        plt.plot(usr, [1] * len(usr), c="r", marker='|', ls="")
        plt.axis([mindate, maxdate, 0, 2])
        plt.grid()


        plt.subplot(313)
        plt.plot(timeline, [1] * len(timeline), c="r", marker='|', ls="")
        plt.axis([0, 10080, 0, 2])
        plt.xticks(np.arange(7) * 1440, ["Mon", "Tue", "Wed", "Thur", "Fri", "Sat", "Sun"])
        plt.grid()

        plt.show()
cdf([users_timelines,P.dates_to_minutes()])

