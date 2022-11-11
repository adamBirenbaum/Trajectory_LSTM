
import numpy as np
import os
from outputs_lstm import Output
import matplotlib.pyplot as plt
from scipy import interpolate
import sys



def interp_features(data, dt, i):

	

	time = data[:,0]
	time_new = np.arange(time[0],time[-1],dt)

	interp_data = np.array([interpolate.interp1d(time, data[:,i])(time_new) for i in range(1,14)]).T
	ids = np.ones((interp_data.shape[0],1))*i

	return np.hstack((ids, np.expand_dims(time_new,axis=1),interp_data))




def process_data(raw_dir, n, dt):

	runs_per_file = 10000
	proc_dir = '/home/adambirenbaum/Documents/SEG/Trajectory_ML/outputs/Processed/Output4'

	output = Output(raw_dir)

	out_inds = output.nums

	nfiles = n // runs_per_file
	remaining = n % runs_per_file
	

	# num runs total
	total_runs = 0
	# num runs in file
	current_file = 0
	# file number with 10000 runs
	ifile = 0
	# empty_array
	array_to_write = []
	while total_runs < n:

		amtDone = current_file / runs_per_file

		sys.stdout.write("\rProgress: [{0:50s}] {1:.1f}%".format('#' * int(amtDone * 50), amtDone * 100))

		file_ind = out_inds[total_runs]
		data = np.loadtxt(os.path.join(raw_dir, 'Outputs_run_{:07d}.txt'.format(file_ind)))

		array_to_write.append(interp_features(data, dt, total_runs))


		

		total_runs += 1
		current_file += 1

		if current_file == runs_per_file:
			filename = os.path.join(proc_dir, 'processed_{:02d}.txt'.format(ifile))
			np.savetxt(filename, np.concatenate(array_to_write))

			ifile += 1
			array_to_write = []
			current_file = 0


	# partially done file (not 10,000 runs) still need to write

	if current_file != 0:
		filename = os.path.join(proc_dir, 'processed_{:02d}.txt'.format(ifile))
		np.savetxt(filename, np.concatenate(array_to_write))


	breakpoint()


if __name__ == '__main__':

	
	dt = 0.25

	raw_dir = '/home/adambirenbaum/Documents/SEG/Trajectory_ML/outputs/Raw/Output4'

	n = int(len(os.listdir(raw_dir))/3)
	
	process_data(raw_dir, n, dt)