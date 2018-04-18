import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM, ConvLSTM2D, GRU, Conv1D
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

def get_dataset(filename='text.txt'):

	# load ascii text and covert to lowercase
	raw_text = open(filename).read()
	raw_text = raw_text.lower()
	# create mapping of unique chars to integers
	chars = sorted(list(set(raw_text)))
	char_to_int = dict((c, i) for i, c in enumerate(chars))
	int_to_char = dict((i, c) for i, c in enumerate(chars))
	# summarize the loaded data
	n_chars = len(raw_text)
	n_vocab = len(chars)

	# Here you have the characters (the vocabulary) and the text
	print "Total Characters: ", n_chars
	print "Total Vocab: ", n_vocab


	# prepare the dataset of input to output pairs encoded as integers

	# The idea to be performed is to try to predict the next character from the previous N characters (N = seq_length)
	seq_length = 100
	dataX = []
	dataY = []
	for i in range(0, n_chars - seq_length, 1):
		seq_in = raw_text[i:i + seq_length]
		seq_out = raw_text[i + seq_length]
		dataX.append([char_to_int[char] for char in seq_in])
		dataY.append(char_to_int[seq_out])

	n_patterns = len(dataX)
	print "Total Patterns: ", n_patterns
	# reshape X to be [samples, size_of_input_data, features]
	X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
	# normalize
	X = X / float(n_vocab)
	# one hot encode the output variable
	y = np_utils.to_categorical(dataY)

	return dataX, X, y, char_to_int, int_to_char, n_vocab


def get_model(X, y):

	# define the LSTM model
	model = Sequential()
	model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
	model.add(Dropout(0.2))
	model.add(LSTM(256))
	model.add(Dropout(0.2))
	model.add(Dense(y.shape[1], activation='softmax'))
	return model

def get_model2(X, y):

	# define the LSTM model
	model = Sequential()
	model.add(GRU(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
	model.add(Dropout(0.2))
	model.add(GRU(256))
	model.add(Dropout(0.2))
	model.add(Dense(y.shape[1], activation='softmax'))
	return model

