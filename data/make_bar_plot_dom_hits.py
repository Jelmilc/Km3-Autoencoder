# -*- coding: utf-8 -*-
"""
Make a quick plot of the statistics of DOM hits for the channel ID AE.
"""
import matplotlib
matplotlib.rcParams.update({'font.size': 16})
import numpy as np
import matplotlib.pyplot as plt

file = "elec-CC_and_muon-CC_xyzc_train_1_to_480_shuffled_0_statistics_fraction_0.02.npy"
file2 = "elec-CC_and_muon-CC_c_event_train_1_to_240_shuffled_0_statistics_fraction_1.npy"
max_hits = 6

array = np.load(file)
array2=np.load(file2)

total_doms = np.sum(array)
total_doms2=np.sum(array2)

def get_for_plot(array, max_hits, total_doms):
    for_plot = np.zeros(max_hits+1)
    hits_array = np.zeros(max_hits+1)
    
    for hits in range(max_hits):
        for_plot[hits]=100*array[hits]/total_doms
        hits_array[hits]=hits
    for_plot[-1]=100*np.sum(array[max_hits:])/total_doms
    hits_array[-1]=max_hits
    
    return for_plot, hits_array

for_plot, hits_array = get_for_plot(array, max_hits, total_doms)
for_plot2, hits_array2 = get_for_plot(array2, max_hits, total_doms2)

xtick_labels = list(range(max_hits))+[str(max_hits)+" or more",]
xticks = np.arange(0,max_hits+1)

y_ticks = [0.1, 1, 10, 100]
ytick_labels = y_ticks

fig=plt.figure(figsize=(9,6))
plt.grid(zorder=-10)

#plt.plot(hits_array, for_plot, "o-")
#plt.semilogy(hits_array, for_plot, "o-")
plt.bar(hits_array-0.15, for_plot, width=0.3,zorder=10, label="Whole file")
plt.bar(hits_array2+0.15, for_plot2, width=0.3,zorder=10, label="With cut")

plt.yscale("log")
plt.legend()
plt.xticks(xticks, xtick_labels)
plt.yticks(y_ticks, ytick_labels)
plt.suptitle("Total number of hits per DOM")
plt.xlabel("Hits per DOM")
plt.ylabel("Percentage of total DOMs")
plt.show()