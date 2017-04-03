from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from keras import optimizers
from keras.models import load_model
import os

def fit(x_train, y_train, learning_rate, batch_size, hidden1, hidden2, hidden3,
				   epochs, savepath, seedNum):
	# define model
	model = Sequential()
	model.add(Dense(40, input_dim = np.shape(x_train)[1], activation='sigmoid'))
	model.add(Dense(20, activation='sigmoid'))
	model.add(Dense(20, activation='sigmoid'))
	model.add(Dense(1, activation = 'linear'))
	ADAM = optimizers.adam(lr=learning_rate)
	model.compile(loss='mean_squared_error', optimizer='adam')

	#TODO
	"""
	Future feature:
	one parameter for hidden
		ex: [40, 20, 20, 1]
	rather than hidden1, hidden2, etc
	"""

	# train model
	model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=epochs)

	# save model
	model.save(os.path.join(savepath.format("keras%s.h5" % seedNum)))

def predict(x_test, savepath, seedNum):
	print(np.shape(x_test), np.asarray(x_test).dtype)
	model = load_model(savepath.format("keras%s.h5" % seedNum))
	#model = load_model(savepath.format("kerasModelSave.h5"))
	return model.predict(x_test)