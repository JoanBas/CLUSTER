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
from copy import deepcopy



class MODEL():
    def __init__(self, time=0, increased_duration_factor=1, event_kinds=['views', 'comments'],dataset_name="handson3", mu_increase_factor=1.):
        ######_INITIALIZATION PARAMETERS_#############

        self.M = 2
        self.event_kinds = event_kinds
        self.event_kind_sess_len_reference = self.event_kinds[0]
        self.dataset_name=dataset_name
        self.reference_stop_event = self.event_kinds.index(self.event_kind_sess_len_reference)  # index of the event used to decid when to stop simulating an event
        self.use_time=time
        self.mu_increase_factor=mu_increase_factor

        ##############################################

        reader = csv.reader(open('Files/in_sess_multi_combined_' + self.dataset_name + "_"+ "_".join(str(x) for x in self.event_kinds) + '.csv', 'r'))
        self.in_sess = []
        for row in reader:
            parameters = row[:]
            alpha = np.reshape(parameters[0:self.M ** 2], [self.M, self.M]).astype(np.float32)
            beta = np.reshape(parameters[self.M ** 2:2 * self.M ** 2], [self.M, self.M]).astype(np.float32)
            mu = np.reshape(parameters[2 * self.M ** 2:2 * self.M ** 2 + self.M], [self.M]).astype(np.float32)
            self.in_sess = [alpha, beta, mu]

        reader = csv.reader(open('Files/sess_dist_' + self.dataset_name + '.csv', 'r'))
        self.sess_dist = {}
        for row in reader:
            self.sess_dist[row[0]] = np.asarray(row[1:]).astype(np.float32)

        if time:
            reader = csv.reader(open('Files/sess_duration_dist_' + self.dataset_name + '.csv', 'r'))
            self.sess_len_dist = {}
            for row in reader:
                self.sess_len_dist[row[0]] = np.asarray(row[1:]).astype(float)
        else:
            reader = csv.reader(
                open('Files/sess_len_dist_' + self.dataset_name + '_' + self.event_kind_sess_len_reference + '.csv',
                     'r'))
            self.sess_len_dist = {}
            for row in reader:
                self.sess_len_dist[row[0]] = np.asarray(row[1:]).astype(int)



    def simulate(self, plot=1, plotAll=0):         # Loop per triar linici de les sessions i omplir aquestes sessions
        P = dataset(dataset_name=self.dataset_name, event_kinds=self.event_kinds)
        users = [P.data[ke] for ke in self.event_kinds]
        users_timelines = [{}, {}]
        N_weeks = int(P.timedifference.days / 7.)
        n_sessions=0
        for usr_id in P.IDs:
            ti = 0.
            t = 0.
            timeline = [np.empty([1, 0]) for i in range(self.M)]
            for week in range(N_weeks):
                for hour_val in self.sess_dist[usr_id]:
                    if t <= ti:
                        rand = random.random()
                        t += random.random() * 60
                    elif t > ti + 60:
                        rand = 1
                    elif t > ti and t < ti + 60:
                        rand = random.random() * (ti + 60 - t) / 60
                        t += random.random() * (ti + 60 - t)
                    else:
                        "ALGO PASAAA"
                    if hour_val > rand:
                        sess, t = self.make_session(usr_id,t)
                        n_sessions+=1
                        for event_kind in range(self.M):
                            timeline[event_kind] = np.append(timeline[event_kind], sess[event_kind])
                    ti += 60.
                    if t < ti: t = ti

            users_timelines[0][usr_id] = timeline[0]
            users_timelines[1][usr_id] = timeline[1]

            if plotAll:
                mindate = datetime.strptime('3/11/2014 00:00:00', '%d/%m/%Y %H:%M:%S')
                maxdate = datetime.strptime('9-11-2014 23:59:59', '%d-%m-%Y %H:%M:%S')
                plt.figure(1)
                plt.subplot(311)
                plt.plot(range(24 * 7), self.sess_dist[usr_id])
                plt.xticks(np.arange(7) * 24, ["Mon", "Tue", "Wed", "Thur", "Fri", "Sat", "Sun"])
                plt.axis([0, 24 * 7, 0, 0.5])
                plt.grid()

                plt.subplot(312)
                timeline1 = users[0][usr_id]
                timeline2 = users[1][usr_id]
                plt.plot(timeline1, [1] * len(timeline1), c="b", marker='|', ls="")
                plt.plot(timeline2, [1] * len(timeline2), c="r", marker='|', ls="")
                plt.axis([mindate, maxdate, 0, 2])
                plt.grid()

                plt.subplot(313)
                plt.plot(timeline[1], [1] * len(timeline[1]), c="b", marker='|', ls="")
                plt.plot(timeline[0], [1] * len(timeline[0]), c="r", marker='|', ls="")
                plt.axis([0, 10080, 0, 2])
                plt.xticks(np.arange(7) * 1440, ["Mon", "Tue", "Wed", "Thur", "Fri", "Sat", "Sun"])
                plt.grid()

                plt.show()

        # cdf([users_timelines[0],P.dates_to_minutes(pre_users=P.data['views'])])
        # cdf([users_timelines[1],P.dates_to_minutes(pre_users=P.data['comments'])])

        number_events=[]
        for ek in self.event_kinds:
            #number of real sessions:
            n_real_sessions=0
            for id in self.sess_len_dist.keys(): n_real_sessions+=len(self.sess_len_dist[id])
            se, re= cdf([users_timelines[self.event_kinds.index(ek)], P.dates_to_minutes(pre_users=P.data[ek])], just_number_events=1)
            number_events+=[se,re]
            print "# simulated ", ek, ": ", se, "              # of sessions: ", n_sessions * 4
            print "# real ", ek, ": ", re, "              # of sessions: ", n_real_sessions

        if plot:
            for ek in self.event_kinds:
                cdf([users_timelines[self.event_kinds.index(ek)], P.dates_to_minutes(pre_users=P.data[ek])])

        return number_events

    def make_session(self,usr_id, t):
        n = random.choice(self.sess_len_dist[usr_id])
        if n == 0: return [np.empty([0]), np.empty([0])], t
        alpha, beta, mu = deepcopy(self.in_sess)
        mu[0] *= self.mu_increase_factor
        session = simulate_multi_hawkes(alpha, beta, mu, n, reference_stop_event='none',
                                        time=self.use_time)  # self.reference_stop_event)
        session[0] += t
        session[1] += t

        T = float('-inf')
        for sec in session:
            try:
                val = sec[-1]
            except:
                val = float("-inf")
            T = max(T, val)
        if len(session[-1]) > 20: print len(session[0]), len(session[-1]), n, alpha, beta, mu
        return session, T


if __name__ == "__main__":

    a=MODEL(time=1)
    ne=[]
    for i in range(20):
        ne. append(a.simulate())
    ne=np.asarray(ne)
    print ne
    mean_ne=np.mean(ne,axis=0)
    print mean_ne


