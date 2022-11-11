
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler



def get_slice_from_apogee(run_data, normed_data, half):

	maxi = np.argmax(normed_data[:,2])

	_slice = slice(maxi - half, maxi + half + 1)


	return run_data[_slice], normed_data[_slice,:]

if __name__ == '__main__':

	proc_dir = '/home/adambirenbaum/Documents/SEG/Trajectory_ML/outputs/Processed/Output4'
	feature_dir = '/home/adambirenbaum/Documents/SEG/Trajectory_ML/outputs/Features/Outputs4'

	half_samples = 10
	nnstart = 30
	nn = 49

	feat_fun = get_slice_from_apogee

	proc_files = os.listdir(proc_dir)

	expected_size = half_samples * 2 + 1
	all_normed_list = []
	all_normal_list = []
	total_runs = 0
	for iproc in range(nnstart,nn):
		
		proc_file = 'processed_{:02d}.txt'.format(iproc)
		print(proc_file)

		data = np.loadtxt(os.path.join(proc_dir, proc_file))


		features = data[:,2:16]
		transformer = RobustScaler().fit(features)
		normalized_features = transformer.transform(features)


		normalized_features[:,6] = (data[:, 8] + 1.0 / 2.0)
		normalized_features[:,7] = (data[:, 9] + 1.0 / 2.0)
		normalized_features[:,8] = (data[:, 10] + 1.0 / 2.0)
		normalized_features[:,9] = (data[:, 11] + 1.0 / 2.0)


		min_ind = int(data[0,0])
		max_ind = int(data[-1,0])

		for i in range(min_ind, max_ind + 1):
			

			irun = data[:,0] == i

			run_data_norm = normalized_features[irun, :]
			run_data = data[irun, :]

			run_features, run_features_norm = feat_fun(run_data,run_data_norm, half_samples)
			
			if run_features.shape[0] != expected_size:
				continue
			total_runs += 1
			all_normed_list.append(run_features_norm)
			all_normal_list.append(run_features)
			
		

	all_feat_normed = np.concatenate(all_normed_list)
	all_feat = np.concatenate(all_normal_list)

	np.savetxt(os.path.join(feature_dir, 'normalized_features{}__{}_{}_13.txt'.format(nnstart, total_runs, expected_size)), all_feat_normed)
	np.savetxt(os.path.join(feature_dir, 'features{}__{}_{}_13.txt'.format(nnstart, total_runs, expected_size)), all_feat)


	breakpoint()







