import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

def get_data(num_classes):
	# the data, split between train and test sets
	(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

	# Scale images to the [0, 1] range
	x_train = x_train.astype("float32") / 255
	x_test = x_test.astype("float32") / 255
	# Make sure images have shape (28, 28, 1)
	x_train = np.expand_dims(x_train, -1)
	x_test = np.expand_dims(x_test, -1)

	x_train_1 = x_train[:,0:14,:]
	x_train_2 = x_train[:,14:28,:]

	x_test_1 = x_test[:,0:14,:]
	x_test_2 = x_test[:,14:28,:]


	print("x_train_1 shape:", x_train_1.shape)
	print("x_train_2 shape:", x_train_2.shape)
	print(x_train.shape[0], "train samples")
	print(x_test.shape[0], "test samples")


	# convert class vectors to binary class matrices
	y_train = keras.utils.to_categorical(y_train, num_classes)
	y_test = keras.utils.to_categorical(y_test, num_classes)

	return (x_train_1, x_train_2, x_test_1, x_test_2, y_train, y_test)

def build_model(num_classes, input_shape):

	model_input_1 = keras.Input(shape=input_shape, name = "left_input")
	model_input_2 = keras.Input(shape=input_shape, name = "right_input")

	conv_left_1 = layers.Conv2D(32, kernel_size = (3,3), activation = "relu")
	conv_right_1 = layers.Conv2D(32, kernel_size = (3,3), activation = "relu")
	x_left = conv_left_1(model_input_1)
	x_right = conv_right_1(model_input_2)

	maxpool_left_1 = layers.MaxPooling2D(pool_size = (2,2))
	maxpool_right_1 = layers.MaxPooling2D(pool_size = (2,2))
	x_left = maxpool_left_1(x_left)
	x_right = maxpool_right_1(x_right)

	conv_left_2 = layers.Conv2D(64, kernel_size = (2,2), activation = "relu")
	conv_right_2 = layers.Conv2D(64, kernel_size = (2,2), activation = "relu")
	x_left = conv_left_2(x_left)
	x_right = conv_right_2(x_right)

	maxpool_left_2 = layers.MaxPooling2D(pool_size = (2,2))
	maxpool_right_2 = layers.MaxPooling2D(pool_size = (2,2))
	x_left = maxpool_left_2(x_left)
	x_right	= maxpool_right_2(x_right)

	flatten_left = layers.Flatten()
	flatten_right = layers.Flatten()
	x_left = flatten_left(x_left)
	x_right = flatten_right(x_right)

	dropout_left_1 = layers.Dropout(0.5)
	dropout_right_1 = layers.Dropout(0.5)
	x_left = dropout_left_1(x_left)
	x_right = dropout_right_1(x_right)

	dense_left_1 = layers.Dense(64, activation = "relu")
	dense_right_1 = layers.Dense(64, activation = "relu")
	collab_left = dense_left_1(x_left)
	collab_right = dense_right_1(x_right)

	left_concat = layers.concatenate([x_left, collab_right])
	right_concat = layers.concatenate([x_right, collab_left])

	dense_left_2 = layers.Dense(num_classes, activation = "softmax", name = "Output_Left")
	dense_right_2 = layers.Dense(num_classes, activation = "softmax", name = "Output_Right")
	x_left = dense_left_2(left_concat)
	x_right = dense_right_2(right_concat)

	#left_model = keras.Model(model_input_1, x_left, name = "Left_Model")
	#right_model = keras.Model(model_input_2, x_right, name = "Right_Model")

	final_model = keras.Model(inputs = [model_input_1, model_input_2], outputs = [x_left, x_right], name = "Final_Model")


	return(final_model)

def main():
	num_classes = 10
	input_shape = (14, 28, 1)

	x_train_1, x_train_2, x_test_1, x_test_2, y_train, y_test = get_data(num_classes)
	model = build_model(num_classes, input_shape)

	model.summary()
	keras.utils.plot_model(model, "model.png")

	batch_size = 128
	epochs = 10

	model.compile(loss=["categorical_crossentropy", "categorical_crossentropy"], optimizer="adam", metrics=["accuracy"])
	model.fit([x_train_1, x_train_2], [y_train, y_train], batch_size=batch_size, epochs=epochs, validation_split=0.1)



	score = model.evaluate([x_test_1, x_test_2], [y_test, y_test], verbose=1)
	print(score)
	print("Test loss:", score[0])
	print("Test accuracy:", score[1])

	# Test Loss: 0.10224977135658264
	# Left Test Loss: 0.04778612405061722
	# Right Test Loss: 0.054463647305965424
	# Left Test Accuracy: 0.9848999977111816
	# Right Test Accuracy: 0.9830999970436096]

if __name__ == "__main__":
	main()