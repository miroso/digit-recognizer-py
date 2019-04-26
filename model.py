# Keras using tensorflow backend

# import the necessary packages
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.models import Sequential


class MiniVGGNet:
    """
    The model is inspired by its bigger brother, VGGNet, and has similar characteristics:
    * Only using 3×3 CONV filters
    * Stacking multiple CONV layers before applying a max-pooling operation
    """
    @staticmethod
	def build(width, height, depth, output_classes):		
		# initialize the model along with the "channels last" ordering since our backend is TensorFlow
        model = Sequential()
		inputShape = (height, width, depth)
		chanDim = -1
        
        # first CONV => RELU => CONV => RELU => POOL layer set
		model.add(Conv2D(32, (3, 3), padding="same",
			input_shape=inputShape))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Conv2D(32, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))
 
		# second CONV => RELU => CONV => RELU => POOL layer set
		model.add(Conv2D(64, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Conv2D(64, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))
 
		# first (and only) set of FC => RELU layers
		model.add(Flatten())
		model.add(Dense(512))
		model.add(Activation("relu"))
		model.add(BatchNormalization())
		model.add(Dropout(0.5))
 
		# softmax classifier
		model.add(Dense(output_classes))
		model.add(Activation("softmax"))
 
		# return the constructed network architecture
		return model


# todo:
# Add an image of the miniVGGnet
# Add stridenet from here: https://www.pyimagesearch.com/2018/12/31/keras-conv2d-and-convolutional-layers/
# is there an alternative to softmax? Is there a way how to handle 51% vs 49%