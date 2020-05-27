import numpy as np
import time
import pickle
import matplotlib.pyplot as plt

list1 = np.linspace(1,100,10)
list2 = np.linspace(1,99,10)
list3 = np.linspace(1,98,10)
list4 = np.linspace(1,97,10)
list5 = np.linspace(1,96,10)
list6 = np.linspace(1,95,10)

list = list1, list2, list3, list4, list5, list6

new_list = []
for element in list:
    new_list.append(np.mean(element))

print(new_list)

xlist = np.linspace(1,10,10), np.linspace(1,10,10), np.linspace(1,10,10), np.linspace(1,10,10), np.linspace(1,10,10), np.linspace(1,10,10),
plt.figure()
plt.plot(new_list)
plt.show()