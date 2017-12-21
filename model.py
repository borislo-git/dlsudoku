import tensorflow as tf
import configparser

class SudokuSolverNetwork(object):
	'''
	Pointer Network

	RNN parts are based on class project implementation

	The output of this network are the a list of indicies corresponding to an element in the input
	'''

	def __init__(self, configFile):
		'''
		Read in paramters from the config file
		'''

		self.loss       = 0
		self.globalStep = tf.Variable(0, trainable=False)
		self.readConfig(configFile)

	def makeGraph(self):
		'''
		makes the graph
		'''
		with tf.variable_scope('Pointer_Network'):
			self.makePlaceholders()

			self.makeCNN()

			self.makeOptimizer()

	def readConfig(self, configFile):
		'''
		Use config parser to get model parameters
		'''

		self.cparser = configparser.ConfigParser()
		self.cparser.read(configFile)
		self.batchSize     = self.cparser.getint('INPUTS', 'BATCH_SIZE')

		self.kernelSize       = self.cparser.getint('CNN', 'KERNEL_SIZE')
		self.numDilationLayer = self.cparser.getint('CNN', 'NUM_DILATION_LAYERS')
		self.hiddenSize          = self.cparser.getint('CNN', 'HIDDEN_SIZE')

		self.l2Reg       = self.cparser.getfloat('TRAIN', 'L2_REG')
		self.dropoutRate = self.cparser.getfloat('TRAIN', 'DROPOUT_RATE')
		self.clipNorm    = self.cparser.getfloat('TRAIN', 'CLIP_NORM_THRESHOLD')

	def makePlaceholders(self):

		with tf.variable_scope('Placeholders'):
			self.train    = tf.placeholder(tf.bool)

			# batch size x 9 x 9 x 1
			self.rawInputs = tf.placeholder(tf.float32, [None, 9, 9, 1])

			# one hot encoding of the solutions
			self.targets   = tf.placeholder(tf.float32, [None, 9, 9, 9])


	def CNNLayer(self, inputs, filters, dilation, name, residual=False, kernelSize=[2,2]):
		'''
		Generates a CNN layer
		'''
		gateName = 'gate_'+str(name)
		filtName = 'filter_'+str(name)

		# apply dropout to input layer only
		if name == 0:
			inputs = tf.layers.dropout(inputs   = inputs,
									   rate     = self.dropoutRate,
									   training = self.train)

		# gated activation unit
		gate = tf.layers.conv2d(inputs        = inputs,
		                        filters       = filters,
		                        kernel_size   = kernelSize,
		                        dilation_rate = dilation,
		                        padding       = 'same',
		                        activation    = tf.sigmoid,
		                        trainable     = True,
		                        name          = gateName)

		filt = tf.layers.conv2d(inputs        = inputs,
		                        filters       = filters,
		                        kernel_size   = kernelSize,
		                        dilation_rate = dilation,
		                        padding       = 'same',
		                        activation    = tf.tanh,
		                        trainable     = True,
		                        name          = filtName)
		out = gate * filt

		# batch norm
		outputs = tf.contrib.layers.batch_norm(inputs=out, 
		                                   decay=0.99,
		                                   center=True, 
		                                   scale=True, 
		                                   activation_fn=None, 
		                                   updates_collections=None,
		                                   is_training=self.train,
		                                   zero_debias_moving_mean=True,
										   fused=True)

		# residual does not add the input context vector which is concatenated after the original outputs of previous layer
		if residual:
			return inputs + out
		else:
			return out

	def makeCNN(self):
		'''
		Set up dilated CNN
		'''

		with tf.variable_scope('Dilated_CNN'):

			self.conv = [self.rawInputs]

			# make the other layers
			factors = [1, 2, 4, 1, 1 ,1, 1, 1]
			for layerNum in range(0, self.numDilationLayer):
				useRes = (layerNum!=0)
				self.conv.append(self.CNNLayer(self.conv[-1], 
									kernelSize = [self.kernelSize,self.kernelSize], 
									filters    = self.hiddenSize, 
									dilation   = factors[layerNum], 
									residual   = useRes, 
									name       = layerNum))

			# depthwise last layer to make unscaled logits
			self.conv.append(self.CNNLayer(self.conv[-1], 
							kernelSize = [1,1], 
							filters    = 9, 
							dilation   = 1, 
							residual   = False, 
							name       = self.numDilationLayer))	


			self.loss    += tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=self.targets, logits=self.conv[-1]))

			# output solution as [batch size x 9 x 9]

			self.prediction = tf.argmax(tf.nn.softmax(self.conv[-1]), axis=-1) + 1

	def makeOptimizer(self):
		'''
		Set up the optimizer and also add regularization and control the learning rate
		'''

		# add L2 regularization on kernels
		for v in tf.trainable_variables():
			if 'kernel' in v.name:
				self.loss += self.l2Reg * tf.nn.l2_loss(v)

		starter_learning_rate = 1e-2
		learning_rate         = tf.train.exponential_decay(starter_learning_rate, self.globalStep,
                                           500, 0.96, staircase=True)
		self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

		# gradient norm clipping
		# gradients, variables = zip(*self.optimizer.compute_gradients(self.loss))
		# gradients, _ = tf.clip_by_global_norm(gradients, self.clipNorm)
		# self.trainOp  = self.optimizer.apply_gradients(zip(gradients, variables), global_step=self.globalStep)

		self.trainOp = self.optimizer.minimize(self.loss, global_step=self.globalStep)

	def printVarsStats(self):
		'''
		Print the names and total number of variables in graph
		'''

		numVars = 0
		for v in tf.trainable_variables():
			print(v.name)
			tmp = 1
			for dim in v.shape:
				tmp *= dim.value
			numVars += tmp
		print('Number of variables: '+str(numVars))


if __name__ == '__main__':
	ntr = SudokuSolverNetwork('hyperparams.cfg')
	ntr.makeGraph()
	ntr.printVarsStats()
