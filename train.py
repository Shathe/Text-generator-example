# Larger LSTM Network to Generate Text for Alice in Wonderland
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import utils

X_raw, X_input, y, _, _, _ = utils.get_dataset(filename='text.txt')
model = utils.get_model(X_input, y)
model.compile(loss='categorical_crossentropy', optimizer='adam')

# define the checkpoint
filepath="weights.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
# fit the model
model.fit(X_input, y, epochs=50, batch_size=64, callbacks=callbacks_list)