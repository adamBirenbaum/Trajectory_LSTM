

import glob
from make_model_function import make_model
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from sklearn.model_selection import train_test_split

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf


physical_devices = tf.config.experimental.list_physical_devices('GPU')

config = tf.config.experimental.set_memory_growth(physical_devices[0], True)


class Output:
	def __init__(self, dir_path):
		self.dir = dir_path

		self.get_nums()

		self.get_inputs()



	def get_nums(self):

		input_files = glob.glob(os.path.join(self.dir, 'Inputs_*.txt'))

		nums = [0] * len(input_files)

		for i, input_file, in enumerate(input_files):

			
			nums[i] = int(re.search('Inputs_run_(.*).txt', os.path.basename(input_file)).group(1))

		nums.sort()
		self.nums = nums

	def get_inputs(self):

		inputs = [None] * len(self.nums)
		normed_inputs = [None] * len(self.nums)

		for i, ilabel in enumerate(self.nums):

			input_file = os.path.join(self.dir, 'Inputs_run_{:07d}.txt'.format(ilabel))
			normed_file = os.path.join(self.dir, 'NormInputs_run_{:07d}.txt'.format(ilabel))

			inputs[i] = np.loadtxt(input_file)

			normed_inputs[i] = np.loadtxt(normed_file)

		inputs = np.vstack(inputs)
		normed_inputs = np.vstack(normed_inputs)

		self.inputs = inputs
		self.normed_inputs = normed_inputs


	def get_labels(self, fun, *params):

		labels = [None] * len(self.nums)

		for i, ilabel in enumerate(self.nums):

			output_file = os.path.join(self.dir, 'Outputs_run_{:07d}.txt'.format(ilabel))
			outputs = np.loadtxt(output_file)

			labels[i] = fun(outputs, *params)
			
		labels = np.squeeze(np.vstack(labels))
		self.labels=labels

	def train(self, train_size, model_path, epochs):
	
		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.normed_inputs, self.labels, train_size=train_size)

		model = make_model(model_path)


		batch_size = 256


		tf.random.set_seed(100)

		model.fit(self.X_train, self.y_train, validation_data=(self.X_test, self.y_test), batch_size=batch_size, epochs=epochs)

		self.model = model
		#model_basename = os.path.splitext(os.path.basename(model_path))[0]
		#model.save('../models/{}.h5'.format(model_basename))

	def predict(self):

		self.predict_test = self.model.predict(self.X_test)
		breakpoint()

	def plot_inputs(self):
		#major_keys = ['Motor','Rocket','Nose','Fin','Tail','Flight']
		minor_keys = ['Thrust', 'Burnout', 'NGrains', 'GrainSep', 'GrainDen', 'NozzleRad', 'ThroatRad', 'Radius', 'Mass', 'InertiaI', 'InertiaZ', 'DistNozzle', 'DistProp', 'Length', 'DistCM', 'NFins', 'Span', 'RootChord', 'TipChord', 'DistCM', 'TopRad', 'BotRad', 'Length', 'DistCM',	'Incline', 'Heading']


		fig, ax = plt.subplots(5,6, figsize=(15,15))
		ax = ax.flatten()
		for i in range(self.inputs.shape[1]):

			ax[i].hist(self.inputs[:,i],bins=50)
			ax[i].set_title(minor_keys[i])

		fig2, ax2 = plt.subplots(5,6, figsize=(15,15))
		ax2 = ax2.flatten()
		for i in range(self.inputs.shape[1]):

			ax2[i].hist(self.normed_inputs[:,i],bins=50)
			ax2[i].set_title(minor_keys[i])


		plt.show()
	



def is_impact_within_circle(data, ymin, ymax):

	return int(data[-1,2] >= ymin and data[-1,2] <= ymax)


if __name__ == '__main__':

	dir_path = '/home/adambirenbaum/Documents/SEG/Trajectory_ML/outputs/Output1'
	model_path = '/home/adambirenbaum/Documents/SEG/Trajectory_ML/models/neural_network1.yaml'

	output = Output(dir_path)
	output.get_labels(is_impact_within_circle, 6000,7000)
	
	output.train(0.7, model_path, 200)

	output.predict()
