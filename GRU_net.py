from lasagne import layers
from lasagne.nonlinearities import rectify, softmax, sigmoid, tanh
import PIL.Image as Image
import pickle
import random
import numpy as np
import theano
import theano.tensor as T
import lasagne

floatX = theano.config.floatX

class BaseNetwork:
	
	def say_name(self):
		return "unknown"
	
	
	def save_params(self, file_name, epoch, **kwargs):
		with open(file_name, 'wb+') as save_file:
			pickle.dump(
				obj = {
					'params' : [x.get_value() for x in self.params],
					'epoch' : epoch, 
				},
				file = save_file,
				protocol = -1
			)

	def load_state(self, file_name):
		print ("####### Cargando epoch:  %s" % file_name)
		epoch = 0
		with open(file_name, 'rb+') as load_file:
			dict = pickle.load(load_file)
			loaded_params = dict['params']
			for (x, y) in zip(self.params, loaded_params):
				x.set_value(y)
			epoch = dict['epoch']
		return epoch


	def get_batches_per_epoch(self, mode):
		if (mode == 'train' or mode == 'predict_on_train'):
			return len(self.train_list_raw) / self.batch_size
		elif (mode == 'test' or mode == 'predict'):
			return len(self.test_list_raw) / self.batch_size
		else:
			raise Exception("unknown mode")
	
	
	def step(self, batch_index, mode):
		
		if (mode == "train"):
			data, answers = self.read_batch(self.train_list_raw, batch_index, mode)####
			theano_fn = self.train_fn
		elif (mode == "test" or mode == "predict"):
                        data, answers = self.read_batch(self.test_list_raw, batch_index, mode)
                        theano_fn = self.test_fn
		elif (mode == "predict_on_train"):
			data, answers = self.read_batch(self.train_list_raw, batch_index, mode)
			theano_fn = self.test_fn
		else:
			raise Exception("unrecognized mode")
		
		ret = theano_fn(data, answers)
		return {"prediction": ret[0],
				"answers": answers,
				"current_loss": ret[1],
				"log": "",
				} 

class Network(BaseNetwork):
    
    def __init__(self, train_list_raw, test_list_raw, png_folder, batch_size, l2, mode, rnn_num_units, **kwargs):
        
        self.train_list_raw = train_list_raw
        self.test_list_raw = test_list_raw
        self.png_folder = png_folder
        self.batch_size = batch_size
        self.l2 = l2
        self.mode = mode
        self.num_units = rnn_num_units
                
        self.input_var = T.tensor3('input_var')
        self.answer_var = T.ivector('answer_var')
        
        ### Arquitectura de la red gru

        example = np.random.uniform(size=(self.batch_size, 768, 256), low=0.0, high=1.0).astype(np.float32) 
        answer = np.random.randint(low=0, high=176, size=(self.batch_size,))

        # Capa de entrada
       
        network = layers.InputLayer(shape=(None, 768, 256), input_var=self.input_var)
        
        # Capa GRU:

        network = layers.GRULayer(incoming=network, num_units=self.num_units, only_return_final=True)
                
        # Ultima Capa de la red

        network = layers.DenseLayer(incoming=network, num_units=122, nonlinearity=softmax)
        
        self.params = layers.get_all_params(network, trainable=True)
        self.prediction = layers.get_output(network)
        
        self.loss_ce = lasagne.objectives.categorical_crossentropy(self.prediction, self.answer_var).mean()
        if (self.l2 > 0):
            self.loss_l2 = self.l2 * lasagne.regularization.regularize_network_params(network, 
                                                                    lasagne.regularization.l2)
        else:
            self.loss_l2 = 0
        self.loss = self.loss_ce + self.loss_l2
        
        updates = lasagne.updates.momentum(self.loss, self.params, learning_rate=0.0005)
        
        if self.mode == 'train':
            self.train_fn = theano.function(inputs=[self.input_var, self.answer_var], 
                                            outputs=[self.prediction, self.loss],
                                            updates=updates)
        
        self.test_fn = theano.function(inputs=[self.input_var, self.answer_var],
                                       outputs=[self.prediction, self.loss])
    
    
    def read_batch(self, data_raw, batch_index, mode):

        start_index = batch_index * self.batch_size
        end_index = start_index + self.batch_size
        
        data = np.zeros((self.batch_size, 768, 256), dtype=np.float32)
        answers = []
        
        for i in range(start_index, end_index):
            name = data_raw[i].split(',')[0]
            answers.append(str(data_raw[i].split(',')[1]))
            
            if mode == 'train':
                path = self.png_folder + name + ".png"     
                im = Image.open(path)
                data[i - start_index, :, :] = np.transpose(np.array(im).astype(np.float32) / 256.0)

            if mode == 'test':
                path = "/home/kevin/Luciano/Spoken-language-identification-master/test_png_images/" + name + ".png"
                im = Image.open(path)
                data[i - start_index, :, :] = np.transpose(np.array(im).astype(np.float32) / 256.0)

            if mode == 'predict':
                path = "/home/kevin/Luciano/Spoken-language-identification-master/test_png_images/" + name + ".png"
                im = Image.open(path)
                data[i - start_index, :, :] = np.transpose(np.array(im).astype(np.float32) / 256.0)

        answers = np.array(answers, dtype=np.int32)
        return data, answers

    def say_name(self):
        return "gru_net"
