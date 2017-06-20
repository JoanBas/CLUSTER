from datetime import datetime
from matplotlib import pyplot as plt
import numpy as np
import os



class dataset():
    def __init__(self,dataset_name='handson3'):
        self.mindate = datetime.strptime('1/11/2014 01:53:00', '%d/%m/%Y %H:%M:%S')
        self.maxdate = datetime.strptime('1-12-2014 00:00:00', '%d-%m-%Y %H:%M:%S')
        self.timedifference= self.maxdate-self.mindate
        self.dataset_name=dataset_name

        def to_sec(x):
            return x.total_seconds()
        self.to_sec= np.vectorize(to_sec)
        self.users_visits=self.get_data("/views.csv")
        self.platform_views=self.get_data_combined("/views.csv")

        #self.users_visits=self.get_data("/user_viewed_design_owner_date.csv")
        #self.users_comment=self.get_data("/user_design_edit_owner_date.csv")
        #self.users_edit=self.get_data("/user_design_edit_owner_date.csv")
        self.IDs = self.most_active_users(self.users_visits, False)

        #self.plot_activity_timeline([self.users_visits])
        #self.plot_which_days(self.users_visits)
        #self.modes()
        #self.day_distribution(self.users_visits)
        #self.average_hour_distribution(self.users_visits,plot=0)

    def get_data(self, filename):
        # interval de dates en les que afagem dades.
        # passem les dades del fitxer a un diccionari on les keys son les id
        # dels usuaris i els continguts son llistes de les dates en les q
        # han interaccionat.
        directori = os.path.dirname(os.path.realpath(__file__))
        directori = os.path.abspath(os.path.join(directori, os.pardir) + '/DADES/'+self.dataset_name)
        file = open(directori + filename, "rt")
        next(file)
        users={}
        activity = []
        acuser = []
        for line in file:
            w = line.split(";")
            user = w[0]
            acuser.append(int(user))
            action = datetime.strptime(w[3][:-1], '%Y-%m-%d %H:%M:%S')
            if action > self.maxdate or action < self.mindate: continue
            if not user in users.keys():
                users[user] = [action]
            else:
                users[user].append(action)
        file.close()
        return users

    def get_data_combined(self,filename):

        directori = os.path.dirname(os.path.realpath(__file__))
        directori = os.path.abspath(os.path.join(directori, os.pardir) + '/DADES/'+self.dataset_name)
        file = open(directori + filename, "rt")
        next(file)
        platform={self.dataset_name:[]}
        for line in file:
            w = line.split(";")
            action = datetime.strptime(w[3][:-1], '%Y-%m-%d %H:%M:%S')
            if action > self.maxdate or action < self.mindate: continue
            platform[self.dataset_name].append(action)
        file.close()
        return platform

    def most_active_users(self, users, plot=False):
        # creem un array (us_ac) on hi ha els codis dels usuaris i el nombre
        # dinteraccions que han fet.
        us_ac = []
        for key in users.keys():
            us_ac.append([int(key), int(len(users[key]))])
        us_ac = np.asarray(us_ac)
        us_ac = us_ac[np.argsort(us_ac[:, 1])]

        if plot:
            plt.hist(us_ac[:, 1],bins=100,  cumulative=True)
            plt.xlabel("Number of events")
            plt.ylabel("Number of users")
            plt.grid(True)
            plt.show()

        most_users=us_ac[-60:, 0]
        return map(str, most_users)

    def prepare(self, ids="none"):

        users = {}
        if ids == "none": ids = self.IDs
        for k in ids:
            ntl = []  # newtimeline
            tl = self.to_sec(np.asarray(self.users_visits[k]) - self.users_visits[k][0]) / 60.  # timeline
            tl_ant = np.roll(tl, 1)
            dtl = tl - tl_ant
            dtl[0] = 1000  # datetime.timedelta(0,7000)
            max_dt = 80  # datetime.timedelta(0,6000)
            j = -1
            for i in range(len(tl)):
                if dtl[i] < max_dt:
                    ntl[j].append(tl[i])
                else:
                    ntl.append([tl[i]])
                    j += 1
            users[k] = ntl
        return users

    def prepare_multivariate(self, ids="none"):
        users = {}
        if ids == "none": ids = self.IDs
        for k in ids:
            user_type= [0 for i in range(len(self.users_visits[k]))]
            user_timeline=[timeline for timeline in self.users_visits[k]]
            try:
                user_type += [1 for i in range(len(self.users_comment[k]))]
                user_timeline += [timeline for timeline in self.users_comment[k]]
                print k, len(self.users_visits[k]), len(self.users_comment[k])

            except: print k, len(self.users_visits[k])

            user = [[x, y] for (x, y) in sorted(zip(user_timeline, user_type))]
            user_type =[y for (x, y) in sorted(zip(user_timeline, user_type))]
            user_timeline.sort()

            ntl = []  # newtimeline
            tl = self.to_sec(np.asarray(user_timeline) - user_timeline[0]) / 60.  # timeline
            tl_ant = np.roll(tl, 1)
            dtl = tl - tl_ant
            dtl[0] = 1000  # datetime.timedelta(0,7000)
            max_dt = 80  # datetime.timedelta(0,6000)
            j = -1
            for i in range(len(tl)):
                if dtl[i] < max_dt:
                    ntl[j][user_type[i]].append(tl[i])
                else:
                    ntl.append([[],[]])
                    ntl[-1][user_type[i]].append(tl[i])
                    j += 1
            users[k] = ntl
        return users

    def keys_to_timeseries(self, keylist):
        ts={}
        for k in keylist:
            ts[k]=self.users_visits[k]
        return ts
    def dates_to_minutes(self, pre_users="none"):
        if pre_users=="none": pre_users=self.users_visits
        users={}
        for id in self.IDs:
            try: users[id]=self.to_sec(np.asarray(pre_users[id]) - pre_users[id][0]) / 60.
            except: users[id]=[]
        return users

    def plot_activity_timeline(self, users_list="none", keys="none"):
        #users list es una llista de diccionaris de users, si nomes
        # posem un diccionari cal posar-lo entre brakets ex: [users]
        if users_list=="none": users_list=[self.users_visits]
        if keys=="none": keys=self.IDs
        c=["red","green","purple","black"]
        i=0
        n = len(keys)
        plt.title("Top 10 activity")
        plt.xlabel("Time")
        plt.ylabel("User id")
        plt.yticks(range(n), keys)
        axes = plt.gca()
        axes.set_ylim([-0.5, n - 0.5])
        for users in users_list:
            usr = []
            yusr = []
            j = 0
            for id in keys:
                try:
                    usr_ = users[id]
                    yusr_ = [j] * len(usr_)

                    usr += usr_
                    yusr += yusr_
                    j += 1
                except:continue
            plt.plot(usr, yusr, c=c[i], marker='|', ls="")
            i+=1

        plt.show()

    def average_weekly_number_of_sessions(self):
        dt=self.timedifference.days

    def average_hour_distribution(self, users, ids="none", plot=1):
        users_hists={}
        if ids=="none": ids=self.IDs
        for id in ids:
            hours=[]
            i=-1
            hind=[]
            dia_=[0]*24
            dant=99
            for date in users[str(id)]:
                h=date.hour
                d=date.day
                hours.append(h)
                if d==dant: hind[i][h]+=1
                else:
                    dant=d
                    dia=dia_[:]
                    hind.append(dia)
                    i+=1
                    hind[i][h] += 1
            hind=np.asarray(hind)
            hind_mean=np.mean(hind,0)
            var=(hind-hind_mean)**2.
            var=(np.mean(var,0))**0.5
            norm=np.sum(hind)/np.shape(hind)[0]
            hist, bins=np.histogram(hours,bins=range(25))
            hist=np.asarray(hist,dtype=float)
            cumhist=np.cumsum(hist)
            hist= hist/norm
            cumhist/=norm
            #plt.bar(np.arange(24),cumhist)
            #plt.colors()
            users_hists[id]=hist
            if plot:
                print hist
                plt.xlabel("Hour of the day")
                plt.ylabel("average # of events")
                plt.bar(np.arange(24),hist)

                mid = np.arange(24)+0.5
                plt.errorbar(mid, hist, yerr=var, fmt='none',ecolor="r")
                plt.xlim([0,24])
                plt.show()
        return users_hists


    def day_distribution(self,users="none", ids="none",plot=1):
        users_hists={}
        if users=="none": users=self.users_visits
        if ids=="none": ids=self.IDs
        for id in ids:
            day=[]
            dia_=[0]*7
            dinw=[[0,0,0,0,0,0,0]]
            i=0
            we=False
            for date in users[str(id)]:
                d=date.weekday()+1
                day.append(d)
                if d==7:
                    we=True
                if d !=7 and we:
                    we=False
                    dia = dia_[:]
                    dinw.append(dia)
                    i += 1
                    dinw[i][d-1] += 1
                else:
                    dinw[i][d-1] += 1

            #dinw = np.asarray(dinw)
            #print dinw
            #dinw_mean = np.mean(dinw)
            #var = (dinw - dinw_mean) ** 2
            #var = (np.mean(var, 0))**0.5

            #plt.hist(day, bins=7, cumulative=-1)
            hist, bins, _ = plt.hist(day, bins=7)
            if plot:
                print hist
                plt.xlabel("Day of the week")
                plt.ylabel("# of events")
                #mid = 0.5 * (bins[1:] + bins[:-1])
                #plt.errorbar(mid, hist, yerr=var, fmt='none')
                plt.show()
            users_hists[id]=hist
        plt.clf()
        return users_hists

    def plot_which_days(self, users):
        for id in self.IDs:
            days=[]
            act=[]
            i=0
            dant=[]
            ndays=self.to_sec(users[str(id)][-1]-users[str(id)][0])/(3600.*24)
            cnt=0
            for date in users[str(id)]:
                d=date.day
                if d==dant:
                    continue
                if d !=dant:
                    cnt+=1
                    days.append(d)
                    act.append(1)
                    dant=d
            print cnt,ndays
            plt.plot(days,act,"|r")
            plt.show()

    def modes(self, users="none",IDlist="none"):
        if users=="none": users=self.users_visits
        if IDlist=="none": IDlist=self.IDs
        print IDlist
        dts=np.empty([0])
        plot_total=True
        for id in IDlist:
            usr=users[id]
            usr_=np.asarray(usr)
            usr_ant_=np.roll(usr_,1)

            dt=usr_-usr_ant_
            dt=dt[1:]

            dt= self.to_sec(dt)
            dt=dt/60.
            #plt.hist(dt,bins=np.logspace(0, 5, 100),cumulative=-1)
            style=0
            if style==1:
                plt.hist(dt,bins=np.logspace(-1, 28, 29,base=1.5))
                plt.yscale('log', nonposy='clip')
                plt.gca().set_xscale("log",base=2)
                plt.xlabel("dt (minutes)")
                plt.ylabel("Number of repetitions")
                plt.show()

            if style==2:
                plt.hist(dt, bins=np.logspace(-1, 5, 60, base=10))
                plt.yscale('log', nonposy='clip')
                plt.gca().set_xscale("log", base=10)
                plt.xlabel("dt (minutes)")
                plt.show()

            if style==3:
                plt.hist(dt, bins=50)#np.logspace(-1, 5, 60, base=10))
                plt.yscale('log', nonposy='clip')
                plt.gca().set_xscale("log", base=24)
                plt.xlabel("dt (minutes)")
                plt.show()
            if plot_total: dts=np.concatenate([dts,dt])

        if plot_total:
            print dts
            np.asarray(dts)
            plt.hist(dts, bins=np.logspace(-1, 28, 29, base=1.5))
            plt.ylabel("Number of repetitions")
            plt.yscale('log', nonposy='clip')
            plt.gca().set_xscale("log", base=2)
            plt.xlabel("dt (minutes)")
            plt.show()

            plt.hist(dts, bins=np.arange(0,20,0.05))
            plt.ylabel("Number of repetitions")
            plt.xlabel("dt (minutes)")
            plt.show()

            plt.hist(dts, bins=np.arange(100,10000,50))
            plt.ylabel("Number of repetitions")
            plt.xlabel("dt (minutes)")
            plt.show()

            decimals=[]
            for i in dts[:3000]:
                decimals.append(i-int(i))
            plt.hist(decimals,bins=60)
            plt.show()

if __name__ == "__main__":
    a=dataset()
    #a.modes(a.users_comment)
    #a.prepare_multivariate()
    #a.modes(users=a.users_edit)
    a.plot_activity_timeline(users_list=[a.users_comment, a.users_visits])
    print a.users_edit
    exit()
    users=a.prepare()
    ht=[]
    for id in a.IDs:
        user=users[id]
        h=[]
        for sess in user:
            h.append(len(sess))
            ht.append(len(sess))
        #plt.hist(h)
        #plt.show()
    h,a,b=plt.hist(ht, bins=100)
    plt.show()
