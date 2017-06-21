from MODEL_multi_combined import MODEL
import numpy as np
import multiprocessing as mtp

def two_options():
    event_pairs=[["views","comments"],["edits","comments"]]
    for event_pair in event_pairs:
        total_means=[]
        for i in range(10):
            print "RONDA", i
            mu_increased_factors=[1.,1.2]
            n_events_total=[]
            for factor in mu_increased_factors:
                n_events=[]
                model=MODEL(time=1, increased_duration_factor=1.,event_kinds=event_pair,mu_increase_factor=factor)
                for j in range(100):
                    n_events.append(model.simulate(plot=0))
                n_events=np.asarray(n_events)
                n_events_total.append(n_events)
            means=[]
            for ne in n_events_total:
                means.append(np.mean(ne,axis=0))
            total_means.append(means)

        print total_means
        file=open("_".join(event_pair)+".csv","wt")
        for means in total_means:
            towrite=[]
            for factor_trial in means:
                towrite+= [factor_trial[0],factor_trial[2]]
            print towrite
            file.write(",".join([str(val) for val in towrite])+"\n")
        file.close()

def single(num,a):
    event_pair=["views","tools"]
    means = []
    stds = []
    for option in range(3):
        if option == 0: mu_increased_factors = [1., 1.]
        elif option == 1: mu_increased_factors = [1.3, 1.]
        elif option == 2: mu_increased_factors = [1., 1.3]
        else: exit()
        total_means=[]
        model = MODEL(time=1, increased_duration_factor=1., event_kinds=event_pair,
                      mu_increase_factor=mu_increased_factors)
        n_events = []
        for i in range(50):
            print "RONDA ", i, "option ", option
            n_events.append(model.simulate(plot=0))
        n_events = np.asarray(n_events)
        print option, n_events
        mean=np.mean(n_events,0)
        std=np.std(n_events,0)
        print mean, std
        means.append(mean)
        stds.append(std)

    print means
    print stds
    np.savetxt("views_tools_mean"+str(num)+".csv", np.asarray(means))
    np.savetxt("views_tools_stds"+str(num)+".csv", np.asarray(stds))


jobs=[]
for num in range(5):
    p=mtp.Process(target=single, args=(num,5))
    jobs.append(p)
    p.start()

