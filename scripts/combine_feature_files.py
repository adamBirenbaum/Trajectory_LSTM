
import numpy as np


file1 = '/home/adambirenbaum/Documents/SEG/Trajectory_ML/outputs/Features/Outputs4/features0__296911_21_13.txt'

file2 = '/home/adambirenbaum/Documents/SEG/Trajectory_ML/outputs/Features/Outputs4/features30__188059_21_13.txt'


file3 = '/home/adambirenbaum/Documents/SEG/Trajectory_ML/outputs/Features/Outputs4/normalized_features0__296911_21_13.txt'

file4 = '/home/adambirenbaum/Documents/SEG/Trajectory_ML/outputs/Features/Outputs4/normalized_features30__188059_21_13.txt'


a = np.loadtxt(file1)
b = np.loadtxt(file2)

c = np.concatenate([a,b])

size = int(c.shape[0] / 21)

np.savetxt('/home/adambirenbaum/Documents/SEG/Trajectory_ML/outputs/Features/Outputs4/features__{}_21_13.txt'.format(size), c)


del a
del b
del c

a = np.loadtxt(file3)
b = np.loadtxt(file4)

c = np.concatenate([a,b])

size = int(c.shape[0] / 21)

np.savetxt('/home/adambirenbaum/Documents/SEG/Trajectory_ML/outputs/Features/Outputs4/normalized_features__{}_21_13.txt'.format(size), c)
