# -*- coding: utf-8 -*-
"""
Make plot of activation functions.
"""

import matplotlib.pyplot as plt
import numpy as np


def softmax(x):
    #for 3 inputs, two of which are 0
    return np.exp(x)/(np.exp(x)+np.exp(0)+np.exp(0))

def sigmoid(x):
    return 1/(np.exp(-x)+1)

#figsize = [6.4,5.5]   
figsize=[13,5.5]
font_size=14

plt.rcParams.update({'font.size': font_size})
plt.subplots(figsize=figsize)


xrange=(-6,6)
x = np.linspace(xrange[0],xrange[1],1000)

plt.plot(x,softmax(x), label="Softmax")
#plt.plot(x, (0.5*np.tanh(x)+0.5), label="tanh")
plt.plot(x, np.maximum(0,x), label="ReLu")
plt.plot(x, sigmoid(x), label="Sigmoid")


plt.xlim(xrange)
plt.ylim(-0.1,1.1)

plt.xticks(np.arange(-6,7,1))

plt.xlabel(r"x")
plt.ylabel(r"f(x)")
plt.grid()

plt.legend()
plt.show()

save_as="../../results/plots/activation_functions.pdf"
plt.savefig(save_as)
print("Saved plot to", save_as)
