
import glob
import hdbscan
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import sys
import umap
import umap.plot


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, Dropout
from tensorflow.compat.v1.keras.layers import CuDNNLSTM

from tensorflow.keras.metrics import mse

physical_devices = tf.config.experimental.list_physical_devices('GPU')

config = tf.config.experimental.set_memory_growth(physical_devices[0], True)




def get_test_features(norm_feat_path, orig_feat_path, train_frac):


	feat_base = os.path.basename(norm_feat_path)
	nruns = int(re.search('.*_([0-9]+)_[0-9]+_13.txt', feat_base).group(1))
	nsteps = int(re.search('.*_[0-9]+_([0-9]+)_13.txt', feat_base).group(1))

	orig_feat = np.loadtxt(orig_feat_path)

	X = np.loadtxt(norm_feat_path)

	X = X.reshape((nruns, nsteps, 13))
	orig_feat = orig_feat.reshape((nruns), nsteps, 15)

	ntrain = int(np.floor(train_frac * nruns))

	test_slice = slice(ntrain,None)
	

	test_inds = orig_feat[test_slice, 0,0]

	test_features = X[test_slice,:,:]

	return test_features, test_inds


def make_encoder(model_file):

	model = tf.keras.models.load_model(model_file)


	encode_end_ind = [i for i, lay_name in enumerate(model.layers) if 'repeat_vector' in lay_name.name][0]

	return tf.keras.models.Sequential(model.layers[:encode_end_ind])



if __name__ == '__main__':




	model_dir = '/home/adambirenbaum/Documents/SEG/Trajectory_ML/outputs/models/iteration1'


	norm_feat_path = '/home/adambirenbaum/Documents/SEG/Trajectory_ML/outputs/Features/normalized_features__172447_21_13.txt'
	orig_feat_path = '/home/adambirenbaum/Documents/SEG/Trajectory_ML/outputs/Features/features__172447_21_13.txt'


	proc_dir = '/home/adambirenbaum/Documents/SEG/Trajectory_ML/outputs/Processed'
	train_frac = 0.9

	min_cluster_sizes = [32, 64, 128, 256, 512]
	n_neighbors = 200

	N_runs_per_file = 10000

	test_features, test_inds = get_test_features(norm_feat_path, orig_feat_path, train_frac)


	model_files = glob.glob(os.path.join(model_dir, '*.h5'))

	read_in_files = {}

	for model_file in model_files:

		print('Model: {}\n'.format(model_file))

		model_file_path = os.path.join(model_dir, model_file)

		model_base = os.path.splitext(os.path.basename(model_file))[0]
		plot_dir = os.path.join(model_dir, 'plots', 'UMAP', model_base)

		os.makedirs(plot_dir, exist_ok=True)

		encoder = make_encoder(model_file_path)

		encoded_list = []

		nrows = test_features.shape[0]
		n_per_call = 1000

		num_iters = nrows  // n_per_call

		for i in range(num_iters):
			_slice = slice(i*n_per_call, (i+1)*n_per_call)
			encoded_list.append(encoder(test_features[_slice,:,:]))

		remaining = nrows % n_per_call
		if  remaining != 0:
			_slice = slice((i+1)*n_per_call, (i+1)*n_per_call + remaining )
			encoded_list.append(encoder(test_features[_slice,:,:]))


		latent_features = np.concatenate(encoded_list)

		for min_cluster_size in min_cluster_sizes:

			print('Min cluster size: {}'.format(min_cluster_size))

			mapper = umap.UMAP(n_neighbors=n_neighbors).fit(latent_features)

			umap_data = mapper.embedding_

			clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)

			cluster_labels = clusterer.fit_predict(umap_data)

			max_label = np.max(cluster_labels)
			fig, ax = plt.subplots(4,2, figsize=(16,10))
			ax = ax.flatten()

			color_palette = sns.color_palette('husl', max_label+1)

			cluster_colors = [color_palette[x] if x >= 0 else (0.5, 0.5, 0.5) for x in cluster_labels]

			cluster_member_colors = [sns.desaturate(x, p) for x, p in
			                         zip(cluster_colors, clusterer.probabilities_)]

			ax[0].scatter(umap_data[:,0], umap_data[:,1], c=cluster_member_colors)
			ax[0].set_title('UMAP')

			


			good_runs = cluster_labels != -1
			
			good_inds = test_inds[good_runs]
			good_colors = np.array(cluster_member_colors)[good_runs]


			for ind, _color in zip(good_inds, good_colors):
				
				nfile = ind // N_runs_per_file
				proc_file = os.path.join(proc_dir, 'processed_{:02d}.txt'.format(int(nfile)))

				if proc_file in read_in_files:
					data = read_in_files[proc_file]
				else:
					data = np.loadtxt(proc_file)
					read_in_files[proc_file] = data

				run_data = data[data[:,0] == ind,:]


				# time vs z
				ax[1].plot(run_data[:,1], run_data[:,4],c=_color)
				# range vs z
				ax[2].plot(run_data[:,3], run_data[:,4],c=_color)
				# time vs vy
				ax[3].plot(run_data[:,1], run_data[:,6],c=_color)
				# time vs vz
				ax[4].plot(run_data[:,1], run_data[:,7],c=_color)
				# time vs e0
				ax[5].plot(run_data[:,1], run_data[:,8],c=_color)
				# time vs e1
				ax[6].plot(run_data[:,1], run_data[:,9],c=_color)
				# time vs w0
				ax[6].plot(run_data[:,1], run_data[:,12],c=_color)



			fig.savefig(os.path.join(plot_dir, 'umap__{}.png'.format(min_cluster_size)),dpi=300)
			plt.close('all')
			


	#encoder.save('../models/encoder_{}.h5'.format(os.path.basename(model_yaml)))


	

	# latent_features = encoder(X[:N, :, :])

	# orig_feat = np.loadtxt(orig_feat_path)
	

	# inds = orig_feat[:N, 0]
	

	# mapper = umap.UMAP(n_neighbors=100).fit(latent_features)

	# umap_data = mapper.embedding_

	# clusterer = hdbscan.HDBSCAN(min_cluster_size=50)

	# cluster_labels = clusterer.fit_predict(umap_data)

	# max_label = np.max(cluster_labels)
	# fig, ax = plt.subplots(1,2, figsize=(16,10))


	
	# color_palette = sns.color_palette('Paired', max_label+1)

	# cluster_colors = [color_palette[x] if x >= 0 else (0.5, 0.5, 0.5) for x in cluster_labels]

	# cluster_member_colors = [sns.desaturate(x, p) for x, p in
	#                          zip(cluster_colors, clusterer.probabilities_)]

	# ax[0].scatter(umap_data[:,0], umap_data[:,1], c=cluster_member_colors)
	# ax[0].set_title('UMAP')

	
	# proc_file = '/home/adambirenbaum/Documents/SEG/Trajectory_ML/outputs/Processed/processed_00.txt'

	# proc_data = np.loadtxt(proc_file)
	


	# ii = 0
	# while ii < N:
	# 	ith_runs = proc_data[:,0] == ii

	# 	if np.any(ith_runs):
	# 		run_data = proc_data[ith_runs,:]

	# 		ax[1].plot(run_data[:,1], run_data[:,4],c=cluster_member_colors[ii])

	# 	ii += 1
	
	# plt.show()