from layer import operation

import numpy as np

class Lenet:
	def function(x,y):
		conv_output1=operation.convolution(x,no_of_filters=6,filter_size=5,stride=1,pad=0)
		pooling_layer1=operation.max_pooling(conv_output1)
		activation_output1=operation.activation(pooling_layer1)


		conv_output2=operation.convolution(activation_output1,no_of_filters=16,filter_size=5,stride=1,pad=0)
		pooling_layer2=operation.max_pooling(conv_output2)
		activation_output2=operation.activation(pooling_layer2)

		flattening_output=operation.flattening(activation_output2)
		fully_connected=operation.forward_propagation(flattening_output,hiddenlayer=120,output=10)
		Softmax=operation.softmax(fully_connected)
		
		#Backpropagation
		error=Softmax-y
		return error
		





		

