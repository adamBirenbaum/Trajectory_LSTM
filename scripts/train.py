


from make_model_function import make_model
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import seaborn as sns
import shutil
import sys
from sklearn.model_selection import train_test_split

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, Dropout
from tensorflow.compat.v1.keras.layers import CuDNNLSTM

from tensorflow.keras.metrics import mse

physical_devices = tf.config.experimental.list_physical_devices('GPU')

config = tf.config.experimental.set_memory_growth(physical_devices[0], True)



def make_good_model(nfeat, n_seq):
	# define model
	model = Sequential()
	# encoder
	#model.add(LSTM(100, activation='relu', input_shape=(n_seq,1)))
	model.add(LSTM(100, input_shape=(n_seq,nfeat),return_sequences=True,activation='tanh'))
	model.add(Dropout(rate=0.2))
	model.add(LSTM(50,return_sequences=False,activation='tanh'))
	# decoder
	model.add(RepeatVector(n_seq))
	#model.add(LSTM(100, activation='relu', return_sequences=True))

	model.add(LSTM(50, return_sequences=True,activation='tanh'))
	model.add(Dropout(rate=0.2))
	model.add(LSTM(100,return_sequences=True,activation='tanh'))
	model.add(Dropout(rate=0.2))
	model.add(TimeDistributed(Dense(nfeat)))
	model.compile(optimizer='adam', loss='mse',metrics='mse')
	return model




def test_models(model_dir):

	model_files = os.listdir(model_dir)

	for model_file in model_files:
		print('testing {}'.format(model_file))
		fullname = os.path.join(model_dir, model_file)
		model = make_model(fullname, (nsteps, nfeat))

	print('\n\n\nPASS\n\n\n')

if __name__ == '__main__':

	norm_feat_path = '/home/adambirenbaum/Documents/SEG/Trajectory_ML/outputs/Features/Outputs4/normalized_features__491720_21_6.txt'
	orig_feat_path = '/home/adambirenbaum/Documents/SEG/Trajectory_ML/outputs/Features/Outputs4/features__491720_21_6.txt'
	

	plot_dir = '/home/adambirenbaum/Documents/SEG/Trajectory_ML/plots'

	model_dir = '/home/adambirenbaum/Documents/SEG/Trajectory_ML/models/iteration2'

	out_model = '/home/adambirenbaum/Documents/SEG/Trajectory_ML/outputs/models'


	
	plot_iter_name = 'normalized_on_10k_data'

	train_size = 0.9
	epochs=100
	batch_size = 256
	nfeat=6

	
	out_model_dir = os.path.join(out_model, os.path.basename(model_dir))
	tensorboard_dir = os.path.join(out_model_dir, 'tensorboard')
	plot_dir = os.path.join(out_model_dir, 'plots')

	
	os.makedirs(out_model_dir,exist_ok=True)
	os.makedirs(tensorboard_dir, exist_ok=True)
	os.makedirs(plot_dir, exist_ok=True)



	X = np.loadtxt(norm_feat_path)


	

	feat_base = os.path.basename(norm_feat_path)
	nruns = int(re.search('.*_([0-9]+)_[0-9]+_6.txt', feat_base).group(1))
	nsteps = int(re.search('.*_[0-9]+_([0-9]+)_6.txt', feat_base).group(1))
	
	X = X.reshape((nruns, nsteps, 6))

	is_nan = np.any(np.isnan(X))
	
	if is_nan:
		delete_ind = np.where(np.isnan(X))[0][0]
		X = np.delete(X,delete_ind,axis=0)

	ntrain = int(np.floor(train_size * nruns))

	#train_slice = slice(0, ntrain)

	X_train = X[:ntrain,:,:]
	X_test = X[ntrain:,:,:]


	#X_train, X_test, y_train, y_test = train_test_split(X, X, train_size=train_size)
	
	model_files = os.listdir(model_dir)

	test_models(model_dir)

	already_done = ['64_32', '256_128']

	for model_file in model_files:
		print(model_file)

		fullname = os.path.join(model_dir, model_file)
		basename = os.path.splitext(model_file)[0]

		# if basename in already_done:
		# 	continue

		
		model = make_model(fullname, (nsteps, nfeat))

		tb_dir = os.path.join(tensorboard_dir, basename)
		if os.path.exists(tb_dir):
			shutil.rmtree(tb_dir)


		tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tb_dir)
		reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
		                              patience=5, min_lr=0.0001)
	

		#model = make_good_model(13, nsteps)
		# print(model.summary())


		history = model.fit(X_train, X_train, epochs=epochs, batch_size=batch_size,  validation_data=(X_test, X_test), callbacks=tensorboard_callback)
		

		model.save(os.path.join(out_model_dir,'{}.h5'.format(basename)))







		yhat = model.predict(X_test)

		orig_feat = np.loadtxt(orig_feat_path)
		orig_feat = orig_feat.reshape((nruns, nsteps, 8))

		if is_nan:
			orig_feat = np.delete(orig_feat, delete_ind, axis=0)

		fig1, ax1 = plt.subplots(figsize=(10,10))

		ax1.plot(history.history['val_loss'], label='validation')
		ax1.plot(history.history['loss'], label='train')
		ax1.grid()
		ax1.legend()
		ax1.set_xlabel('Epochs')
		ax1.set_ylabel('Loss')
		ax1.set_title('Training Loss: {:4.5f}\nValidation Loss: {:4.5f}'.format(history.history['loss'][-1], history.history['val_loss'][-1]))
		fig1.savefig(os.path.join(plot_dir, 'learning_curve__{}.png'.format(basename)))



		feature_plot_dir = os.path.join(plot_dir, 'features_{}'.format(basename))
		os.makedirs(feature_plot_dir,exist_ok=True)

		feature_labels=['x','y','z','phi','theta','psi']


		for j in range(nfeat):

			n_plots = 16

			fig, ax = plt.subplots(4,4, figsize=(16,16))
			ax = ax.flatten()

			for i in range(n_plots):
				
				#actual = orig_feat[i,:,j+2]
				actual = X_test[i, :, j]
				pred = yhat[i,:,j]
				ax[i].plot(actual, '-r', label='truth')
				ax[i].plot(pred, '--b', label='prediction')
				
				ax[i].grid()
				ax[i].set_title('MSE: {:6.7f}'.format(mse(actual, pred).numpy()))

				ax[i].legend()

			fig.suptitle('Feature: {}'.format(feature_labels[j]))
			fig.savefig(os.path.join(feature_plot_dir, 'feature_{}.png'.format(feature_labels[j])))

	
		
		plt.close('all')

	