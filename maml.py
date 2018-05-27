""" Code for the MAML algorithm and network definitions. """
from __future__ import print_function
import numpy as np
import sys
import tensorflow as tf
try:
    import special_grads
except KeyError as e:
    print('WARN: Cannot define MaxPoolGrad, likely already defined for this version of tensorflow: %s' % e,
          file=sys.stderr)

from tensorflow.python.platform import flags
from utils import mse, xent, conv_block, normalize

FLAGS = flags.FLAGS

auto = True
encodeWeight = 0.3



class MAML:
    def __init__(self, dim_input=1, dim_output=1, test_num_updates=5):
        """ must call construct_model() after initializing MAML! """
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.update_lr = FLAGS.update_lr
        self.meta_lr = tf.placeholder_with_default(FLAGS.meta_lr, ())
        self.auto_lr = tf.placeholder_with_default(FLAGS.auto_lr, ())
        
        self.classification = False
        self.test_num_updates = test_num_updates
        self.dim_auto = 2 #This should be able to be arbitrary
        if auto:
            self.real_input = 39 # This is square root of the total (its a kernel)
            #self.real_output = 40#self.dim_output
            self.real_output = 39*39 # This should be the complete dimension out. 
            self.dim_input = 3*self.dim_auto #= 3*self.dim_auto 
            self.dim_output = self.dim_auto
            #This is from each. 
            #if auto: self.dim_input, self.dim_output = self.dim_auto, self.dim_auto #If auto, pass in/out the dimension of the latent (auto_
        if FLAGS.datasource == 'sinusoid':
            self.dim_hidden = [40, 40,40]
            self.loss_func = mse
            self.forward = self.forward_fc
            self.construct_weights = self.construct_fc_weights
        elif FLAGS.datasource == 'omniglot' or FLAGS.datasource == 'miniimagenet':
            self.loss_func = xent
            self.classification = True
            if FLAGS.conv:
                self.dim_hidden = FLAGS.num_filters
                self.forward = self.forward_conv
                self.construct_weights = self.construct_conv_weights
            else:
                self.dim_hidden = [256, 128, 64, 64]
                self.forward=self.forward_fc
                self.construct_weights = self.construct_fc_weights
            if FLAGS.datasource == 'miniimagenet':
                self.channels = 3
            else:
                self.channels = 1
            self.img_size = int(np.sqrt(self.dim_input/self.channels))
        else:
            raise ValueError('Unrecognized data source.')

    def construct_model(self, input_tensors=None, prefix='metatrain_'):
        # a: training data for inner gradient, b: test data for meta gradient
        if input_tensors is None:
            self.inputa1 = tf.placeholder(tf.float32)#,shape=[None,100,2,2])
            self.inputa2 = tf.placeholder(tf.float32)
            self.inputa3 = tf.placeholder(tf.float32)
            self.inputb1 = tf.placeholder(tf.float32)
            self.inputb2 = tf.placeholder(tf.float32)
            self.inputb3 = tf.placeholder(tf.float32)
            self.labela = tf.placeholder(tf.float32)
            self.labelb = tf.placeholder(tf.float32)
        else:
            self.inputa1 = input_tensors['inputa1']
            self.inputa2 = input_tensors['inputa2']
            self.inputa3 = input_tensors['inputa3']
            self.inputb1 = input_tensors['inputb1']
            self.inputb2 = input_tensors['inputb2']
            self.inputb3 = input_tensors['inputb3']
            self.labela = input_tensors['labela']
            self.labelb = input_tensors['labelb']

        with tf.variable_scope('model', reuse=None) as training_scope:
            if 'weights' in dir(self):
                training_scope.reuse_variables()
                weights = self.weights
                auto_weights = self.auto_weights
            else:
                # Define the weights
                self.weights = weights = self.construct_weights()
                self.auto_weights = auto_weights = self.construct_auto_weights()

            # outputbs[i] and lossesb[i] is the output and loss after i+1 gradient updates
            lossesa, outputas, lossesb, outputbs = [], [], [], []
            accuraciesa, accuraciesb = [], []
            num_updates = max(self.test_num_updates, FLAGS.num_updates)
            outputbs = [[]]*num_updates
            lossesb = [[]]*num_updates
            accuraciesb = [[]]*num_updates

            def task_metalearn(inp, reuse=True):
                """ Perform gradient descent for one task in the meta-batch. """
                inputa1,inputa2,inputa3, inputb1,inputb2,inputb3, labela, labelb = inp
                print("Input a: " , inputa1)
                task_outputbs, task_lossesb, auto_losses = [], [], []
                auto_loss = None

                #One for each. 

                # This takes in the input and passes out the latent variables.
                temp_in_a_1 = self.encoder(inputa1,self.auto_weights, reuse=reuse)
                #Then transform it back, and take the loss
                temp_out_a_1 = self.decoder(temp_in_a_1,self.auto_weights, reuse=reuse)
                auto_out_a_1 = temp_out_a_1

                temp_in_a_2 = self.encoder(inputa2,self.auto_weights, reuse=reuse)
                #Then transform it back, and take the loss
                temp_out_a_2 = self.decoder(temp_in_a_2,self.auto_weights, reuse=reuse)
                auto_out_a_2 = temp_out_a_2

                temp_in_a_3 = self.encoder(inputa3,self.auto_weights, reuse=reuse)
                #Then transform it back, and take the loss
                temp_out_a_3 = self.decoder(temp_in_a_3,self.auto_weights, reuse=reuse)
                auto_out_a_3 = temp_out_a_3

               # This takes in the input and passes out the latent variables.
                temp_in_b_1 = self.encoder(inputb1,self.auto_weights, reuse=reuse)
                #Then transform it back, and take the loss
                temp_out_b_1 = self.decoder(temp_in_b_1,self.auto_weights, reuse=reuse)
                auto_out_b_1 = temp_out_b_1

                temp_in_b_2 = self.encoder(inputb2,self.auto_weights, reuse=reuse)
                #Then transform it back, and take the loss
                temp_out_b_2 = self.decoder(temp_in_b_2,self.auto_weights, reuse=reuse)
                auto_out_b_2 = temp_out_b_2

                temp_in_b_3 = self.encoder(inputb3,self.auto_weights, reuse=reuse)
                #Then transform it back, and take the loss
                temp_out_b_3 = self.decoder(temp_in_b_3,self.auto_weights, reuse=reuse)
                auto_out_b_3 = temp_out_b_3


                #print("temp out a: " , temp_out_a)
                la_1 = self.loss_func(temp_out_a_1,inputa1)
                la_2 = self.loss_func(temp_out_a_2,inputa2)
                la_3 = self.loss_func(temp_out_a_3,inputa3)

                lb_1 = self.loss_func(temp_out_b_1,inputb1)
                lb_2 = self.loss_func(temp_out_b_2,inputb2)
                lb_3 = self.loss_func(temp_out_b_3,inputb3)

                auto_loss = lb_1+lb_2+lb_3

                print("Input a1: " , temp_in_a_1)
                inputa=tf.concat([temp_in_a_1, temp_in_a_2,temp_in_a_3],1)
                inputb=tf.concat([temp_in_b_1, temp_in_b_2,temp_in_b_3],1)
                print("Inputa: " , inputa)
                task_outputa = self.forward(inputa, weights, reuse=reuse)  # only reuse on the first iter
                temp_outputa = self.decoder(task_outputa,self.auto_weights,reuse=reuse)
                
                print("Task outputa: " , temp_outputa)
                print("Label a: " , labela)
                task_lossa = self.loss_func(temp_outputa, labela)

                grads = tf.gradients(task_lossa, list(weights.values()))
                if FLAGS.stop_grad:
                    grads = [tf.stop_gradient(grad) for grad in grads]
                gradients = dict(zip(weights.keys(), grads))
                fast_weights = dict(zip(weights.keys(), [weights[key] - self.update_lr*gradients[key] for key in weights.keys()]))
                output = self.forward(inputb, fast_weights, reuse=True)
                
                temp_outputb = self.decoder(output,self.auto_weights, reuse=True)
                output = temp_outputb

                task_outputbs.append(output)
                print("Output: " , output)
                print("Labels: " , labelb)
                task_lossesb.append(self.loss_func(output, labelb))
                print("Num updates is: " , num_updates-1)

                for j in range(num_updates - 1):
                    loss = self.loss_func(self.decoder(self.forward(inputa, fast_weights, reuse=True),self.auto_weights), labela)
                    grads = tf.gradients(loss, list(fast_weights.values()))
                    if FLAGS.stop_grad:
                        grads = [tf.stop_gradient(grad) for grad in grads]
                    gradients = dict(zip(fast_weights.keys(), grads))
                    fast_weights = dict(zip(fast_weights.keys(), [fast_weights[key] - self.update_lr*gradients[key] for key in fast_weights.keys()]))
                    
                    output = self.forward(inputb, fast_weights, reuse=True)
                    output = self.decoder(output,self.auto_weights)
                    
                    task_outputbs.append(output)
                    task_lossesb.append(self.loss_func(output, labelb))

                task_output = [task_outputa, task_outputbs, task_lossa, task_lossesb, auto_loss,auto_out_a_1,auto_out_a_2]
                return task_output

            out_dtype = [tf.float32, [tf.float32]*num_updates, tf.float32, [tf.float32]*num_updates]
       
            out_dtype.extend([tf.float32,tf.float32,tf.float32])

            result = tf.map_fn(task_metalearn, elems=(self.inputa1,self.inputa2,self.inputa3,self.inputb1,self.inputb2,self.inputb3, self.labela, self.labelb), dtype=out_dtype, parallel_iterations=FLAGS.meta_batch_size)
            #In case you want to fetch it. 
            #if auto:
                #auto_losses = [1,2,3,4]
            outputas, outputbs, lossesa, lossesb, autoloss, auto_out_a,auto_out_b  = result
             
            self.outputbs = outputbs
        #print("Meta batch: " , FLAGS.meta_batch_size)
        ## Performance & Optimization

        self.l1_regularizer_e = tf.contrib.layers.l1_regularizer(
           scale=FLAGS.encoderregularize_penal, scope=None
        )
        self.weights1 = tf.trainable_variables() # all vars of your graph
        regularization_penalty_e = tf.contrib.layers.apply_regularization(self.l1_regularizer_e, self.weights1)


        self.l1_regularizer_p = tf.contrib.layers.l1_regularizer(
           scale=FLAGS.predictregularize_penal, scope=None
        )
        self.weights1 = tf.trainable_variables() # all vars of your graph
        regularization_penalty_p = tf.contrib.layers.apply_regularization(self.l1_regularizer_p, self.weights1)
        

        if 'train' in prefix:
            self.auto_losses = tf.reduce_sum(autoloss) / tf.to_float(FLAGS.meta_batch_size)
            self.total_loss1 = total_loss1 = tf.reduce_sum(lossesa) / tf.to_float(FLAGS.meta_batch_size)
            self.total_losses2 = total_losses2 = [tf.reduce_sum(lossesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]
            #self.auto_losses = auto_losses
            # after the map_fn
            self.outputas, self.outputbs = outputas, outputbs
            self.auto_out_a, self.auto_out_b = auto_out_a, auto_out_b

            self.autotrain_op = tf.train.AdamOptimizer(self.auto_lr).minimize(self.auto_losses + regularization_penalty_e)
            self.pretrain_op = tf.train.AdamOptimizer(self.meta_lr).minimize(total_loss1+encodeWeight*self.auto_losses+ regularization_penalty_e) #For pretraining as well. 
            if FLAGS.metatrain_iterations > 0:
                #print("Meta train....")
                optimizer = tf.train.AdamOptimizer(self.meta_lr)
                lossPenal = self.total_losses2[FLAGS.num_updates-1]
                lossPenal = lossPenal + regularization_penalty_e + encodeWeight*self.auto_losses
                self.gvs = gvs = optimizer.compute_gradients(lossPenal)
                self.metatrain_op = optimizer.apply_gradients(gvs)
        else:
            print("Validation, Meta batch size: " , FLAGS.meta_batch_size)
            self.metaval_total_loss1 = total_loss1 = tf.reduce_sum(lossesa) / tf.to_float(FLAGS.meta_batch_size)
            self.metaval_total_losses2 = total_losses2 = [tf.reduce_sum(lossesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]
            #if self.classification:
            #    self.metaval_total_accuracy1 = total_accuracy1 = tf.reduce_sum(accuraciesa) / tf.to_float(FLAGS.meta_batch_size)
            #    self.metaval_total_accuracies2 = total_accuracies2 =[tf.reduce_sum(accuraciesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]

        ## Summaries
        tf.summary.scalar(prefix+'Pre-update loss', total_loss1)
        if self.classification:
            tf.summary.scalar(prefix+'Pre-update accuracy', total_accuracy1)

        for j in range(num_updates):
            tf.summary.scalar(prefix+'Post-update loss, step ' + str(j+1), total_losses2[j])
            if self.classification:
                tf.summary.scalar(prefix+'Post-update accuracy, step ' + str(j+1), total_accuracies2[j])

    ### Network construction functions (fc networks and conv networks)
    def construct_fc_weights(self):
        weights = {}
        weights['w1'] = tf.Variable(tf.truncated_normal([self.dim_input, self.dim_hidden[0]], stddev=0.01))
        weights['b1'] = tf.Variable(tf.zeros([self.dim_hidden[0]]))
        print("Weights w1: " , weights['w1'].shape)
        for i in range(1,len(self.dim_hidden)):
            weights['w'+str(i+1)] = tf.Variable(tf.truncated_normal([self.dim_hidden[i-1], self.dim_hidden[i]], stddev=0.01))
            weights['b'+str(i+1)] = tf.Variable(tf.zeros([self.dim_hidden[i]]))
        weights['w'+str(len(self.dim_hidden)+1)] = tf.Variable(tf.truncated_normal([self.dim_hidden[-1], self.dim_output], stddev=0.01))
        weights['b'+str(len(self.dim_hidden)+1)] = tf.Variable(tf.zeros([self.dim_output]))
        return weights

    # This constructs the autoencoder weights.
    def construct_auto_weights(self):
        weights = {}
        weights['encoder_a'] = tf.Variable(tf.truncated_normal([512,2],stddev=0.01))
        print("Encoder a: " , weights['encoder_a'].shape)
        weights['decoder_a'] = tf.Variable(tf.truncated_normal([2,512],stddev=0.01))
        dtype = tf.float32
        conv_initializer =  tf.contrib.layers.xavier_initializer_conv2d(dtype=dtype)

        channels = 1
        weights['conv1'] = tf.get_variable('conv1', [3, 3, channels, 32], dtype=dtype)
        weights['b1_e'] = tf.Variable(tf.zeros([32]))
        weights['conv2'] = tf.get_variable('conv2', [3, 3, 32, 32], initializer=conv_initializer, dtype=dtype)
        
        weights['b2_e'] = tf.Variable(tf.zeros([32]))
        weights['conv3'] = tf.get_variable('conv3', [3, 3, 32,32], initializer=conv_initializer, dtype=dtype)
        weights['b3_e'] = tf.Variable(tf.zeros([32]))


        weights['dec_t'] = tf.Variable(tf.zeros([1]))

        weights['conv1_t'] = tf.get_variable('conv1_t', [3, 3, 32, 32], initializer=conv_initializer, dtype=dtype)
        weights['b1_t'] = tf.Variable(tf.zeros([32]))
        weights['conv2_t'] = tf.get_variable('conv2_t', [3, 3, 32, 32], initializer=conv_initializer, dtype=dtype)
        
        weights['b2_t'] = tf.Variable(tf.zeros([32]))
        weights['conv3_t'] = tf.get_variable('conv3_t', [3, 3, 1,32], initializer=conv_initializer, dtype=dtype)
        #weights['b3_t'] = tf.Variable(tf.zeros([1]))
        #weights['conv4'] = tf.get_variable('conv4', [k, k, self.dim_hidden, self.dim_hidden], initializer=conv_initializer, dtype=dtype)
        #weights['b4'] = tf.Variable(tf.zeros([32]))
        return weights



    #Take the entire and responds with the small latent. 
    def encoder(self,inp,weights,reuse=False,scope=''):
        #print("Weights: " , weights.keys())
        #weights['encoder_a'] = weights['w1']
        print("Input: " , inp)

        my_inp = tf.reshape(inp,[-1,39,39,1])
        print("My input: " , my_inp)

        conv1 = normalize(tf.nn.conv2d(my_inp,weights['conv1'],[1,2,2,1],"VALID"),activation=tf.nn.relu,reuse=reuse,scope=scope+"0")+weights['b1_e']
        print("Conv", conv1)

        conv2 = normalize(tf.nn.conv2d(conv1,weights['conv2'],[1,2,2,1],"VALID"),activation=tf.nn.relu,reuse=reuse,scope=scope+"1")+weights['b2_e']
        print("Conv2: " , conv2)

        conv3 = normalize(tf.nn.conv2d(conv2,weights['conv3'],[1,2,2,1],"VALID"),activation=tf.nn.relu,reuse=reuse,scope=scope+"2")+weights['b3_e']
        print("Conv3 : " , conv3)
        #conv2 = tf.layers.conv2d(inputs=conv1,filters=32,kernel_size=[3,3],padding="valid",activation=tf.nn.relu,strides=[2,2])
        #conv3 = tf.layers.conv2d(inputs=conv2,filters=32,kernel_size=[3,3],padding="valid",activation=tf.nn.relu,strides=[2,2])
        print("Conv network: " , conv3)
        #os.exit()
        pool2_flat = tf.reshape(conv3, [-1, 512])

        print("Pool flat: " , pool2_flat)
        
        layer_2 = tf.matmul(pool2_flat,weights['encoder_a'])
        print("Lay2: " , layer_2)
        #os.exit()

        #print("Pool 2 fat: " , pool2_flat)
        return layer_2

    #Takes in the small latent and replies with the entire. 
    def decoder(self,inp,weights,reuse=False,scope=''):

        print("InputT: " , inp)

        layer_2 = tf.matmul(inp,weights['decoder_a'])

        ## TODO ADD A WEIGHT IN THIS. 
        print("Layer 2: " , layer_2)
        my_inp = tf.reshape(layer}_2,[-1,4,4,32]) + weights['dec_t']
        
        conv1 = normalize(tf.nn.conv2d_transpose(my_inp,weights['conv1_t'],[FLAGS.update_batch_size,9,9,32],[1,2,2,1],padding="VALID"),activation=tf.nn.relu,reuse=reuse,scope=scope+"0") + weights['b1_t']

        conv2 = normalize(tf.nn.conv2d_transpose(conv1,weights['conv2_t'],[FLAGS.update_batch_size,19,19,32],[1,2,2,1],padding="VALID"),activation=tf.nn.relu,reuse=reuse,scope=scope+"1") + weights['b2_t']

        conv3 = normalize(tf.nn.conv2d_transpose(conv2,weights['conv3_t'],[FLAGS.update_batch_size,39,39,1],[1,2,2,1],padding="VALID"),activation=tf.nn.relu,reuse=reuse,scope=scope+"1")# + weights['b3_t']

        return conv3

    def forward_fc(self, inp, weights, reuse=False):
        print("Weights:", weights.keys())
        #new_in = self.encoder(inp,weights)


        hidden = normalize(tf.matmul(inp, weights['w1']) + weights['b1'], activation=tf.nn.relu, reuse=reuse, scope='0')
        print("Hidden: " , hidden)
        print("Weights w1: " , weights['w1'])
        for i in range(1,len(self.dim_hidden)):
            hidden = normalize(tf.matmul(hidden, weights['w'+str(i+1)]) + weights['b'+str(i+1)], activation=tf.nn.relu, reuse=reuse, scope=str(i+1))
        return tf.matmul(hidden, weights['w'+str(len(self.dim_hidden)+1)]) + weights['b'+str(len(self.dim_hidden)+1)]

    def construct_conv_weights(self):
        weights = {}

        dtype = tf.float32
        conv_initializer =  tf.contrib.layers.xavier_initializer_conv2d(dtype=dtype)
        fc_initializer =  tf.contrib.layers.xavier_initializer(dtype=dtype)
        k = 3

        weights['conv1'] = tf.get_variable('conv1', [k, k, self.channels, self.dim_hidden], initializer=conv_initializer, dtype=dtype)
        weights['b1'] = tf.Variable(tf.zeros([self.dim_hidden]))
        weights['conv2'] = tf.get_variable('conv2', [k, k, self.dim_hidden, self.dim_hidden], initializer=conv_initializer, dtype=dtype)
        weights['b2'] = tf.Variable(tf.zeros([self.dim_hidden]))
        weights['conv3'] = tf.get_variable('conv3', [k, k, self.dim_hidden, self.dim_hidden], initializer=conv_initializer, dtype=dtype)
        weights['b3'] = tf.Variable(tf.zeros([self.dim_hidden]))
        weights['conv4'] = tf.get_variable('conv4', [k, k, self.dim_hidden, self.dim_hidden], initializer=conv_initializer, dtype=dtype)
        weights['b4'] = tf.Variable(tf.zeros([self.dim_hidden]))
        if FLAGS.datasource == 'miniimagenet':
            # assumes max pooling
            weights['w5'] = tf.get_variable('w5', [self.dim_hidden*5*5, self.dim_output], initializer=fc_initializer)
            weights['b5'] = tf.Variable(tf.zeros([self.dim_output]), name='b5')
        else:
            weights['w5'] = tf.Variable(tf.random_normal([self.dim_hidden, self.dim_output]), name='w5')
            weights['b5'] = tf.Variable(tf.zeros([self.dim_output]), name='b5')
        return weights

    def forward_conv(self, inp, weights, reuse=False, scope=''):
        # reuse is for the normalization parameters.
        channels = self.channels
        inp = tf.reshape(inp, [-1, self.img_size, self.img_size, channels])

        hidden1 = conv_block(inp, weights['conv1'], weights['b1'], reuse, scope+'0')
        hidden2 = conv_block(hidden1, weights['conv2'], weights['b2'], reuse, scope+'1')
        hidden3 = conv_block(hidden2, weights['conv3'], weights['b3'], reuse, scope+'2')
        hidden4 = conv_block(hidden3, weights['conv4'], weights['b4'], reuse, scope+'3')
        if FLAGS.datasource == 'miniimagenet':
            # last hidden layer is 6x6x64-ish, reshape to a vector
            hidden4 = tf.reshape(hidden4, [-1, np.prod([int(dim) for dim in hidden4.get_shape()[1:]])])
        else:
            hidden4 = tf.reduce_mean(hidden4, [1, 2])

        return tf.matmul(hidden4, weights['w5']) + weights['b5']


