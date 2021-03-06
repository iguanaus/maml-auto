"""
Hi

Usage Instructions:
    10-shot sinusoid:
        python main.py --datasource=sinusoid --logdir=logs/sine/ --metatrain_iterations=70000 --norm=None --update_batch_size=10

    10-shot sinusoid baselines:
        python main.py --datasource=sinusoid --logdir=logs/sine/ --pretrain_iterations=70000 --metatrain_iterations=0 --norm=None --update_batch_size=10 --baseline=oracle
        python main.py --datasource=sinusoid --logdir=logs/sine/ --pretrain_iterations=70000 --metatrain_iterations=0 --norm=None --update_batch_size=10

    5-way, 1-shot omniglot:
        python main.py --datasource=omniglot --metatrain_iterations=40000 --meta_batch_size=32 --update_batch_size=1 --update_lr=0.4 --num_updates=1 --logdir=logs/omniglot5way/

    20-way, 1-shot omniglot:
        python main.py --datasource=omniglot --metatrain_iterations=40000 --meta_batch_size=16 --update_batch_size=1 --num_classes=20 --update_lr=0.1 --num_updates=5 --logdir=logs/omniglot20way/

    5-way 1-shot mini imagenet:
        python main.py --datasource=miniimagenet --metatrain_iterations=60000 --meta_batch_size=4 --update_batch_size=1 --update_lr=0.01 --num_updates=5 --num_classes=5 --logdir=logs/miniimagenet1shot/ --num_filters=32 --max_pool=True

    5-way 5-shot mini imagenet:
        python main.py --datasource=miniimagenet --metatrain_iterations=60000 --meta_batch_size=4 --update_batch_size=5 --update_lr=0.01 --num_updates=5 --num_classes=5 --logdir=logs/miniimagenet5shot/ --num_filters=32 --max_pool=True

    To run evaluation, use the '--train=False' flag and the '--test_set=True' flag to use the test set.

    For omniglot and miniimagenet training, acquire the dataset online, put it in the correspoding data directory, and see the python script instructions in that directory to preprocess the data.
"""
import csv
import numpy as np
import pickle
import random
import tensorflow as tf

from data_generator import DataGenerator
from maml import MAML
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS

## Dataset/method options
flags.DEFINE_string('datasource', 'sinusoid', 'sinusoid or omniglot or miniimagenet')
flags.DEFINE_integer('num_classes', 5, 'number of classes used in classification (e.g. 5-way classification).')
# oracle means task id is input (only suitable for sinusoid)
flags.DEFINE_string('baseline', None, 'oracle, or None')

## Training options
flags.DEFINE_integer('encoder_iterations', 2000, 'number of auto-training iterations.')

flags.DEFINE_integer('pretrain_iterations', 0, 'number of pre-training iterations.')
flags.DEFINE_integer('metatrain_iterations', 15000, 'number of metatraining iterations.') # 15k for omniglot, 50k for sinusoid
flags.DEFINE_integer('meta_batch_size', 25, 'number of tasks sampled per meta-update')
flags.DEFINE_float('meta_lr', 1e-3, 'the base learning rate of the generator')
flags.DEFINE_float('auto_lr', 1e-3, 'the base learning rate of the auto encoder (for pretraining)')
flags.DEFINE_integer('update_batch_size', 5, 'number of examples used for inner gradient update (K for K-shot learning).')
flags.DEFINE_float('update_lr', 1e-3, 'step size alpha for inner gradient update.') # 0.1 for omniglot
flags.DEFINE_integer('num_updates', 1, 'number of inner gradient updates during training.')

flags.DEFINE_float('encoderregularize_penal', 1e-8, 'Regularization penalty encoding') # For Encoding training
flags.DEFINE_float('predictregularize_penal', 1e-8, 'Regularization penalty prediction') # For Normal training

flags.DEFINE_bool('limit_task', True, 'if True, limit the # of tasks shown')
flags.DEFINE_integer('limit_task_num', 4, 'if True, limit the # of tasks shown')


## Model options
flags.DEFINE_string('norm', 'None', 'batch_norm, layer_norm, or None')
flags.DEFINE_integer('num_filters', 64, 'number of filters for conv nets -- 32 for miniimagenet, 64 for omiglot.')
flags.DEFINE_bool('conv', True, 'whether or not to use a convolutional network, only applicable in some cases')
flags.DEFINE_bool('max_pool', False, 'Whether or not to use max pooling rather than strided convolutions')
flags.DEFINE_bool('stop_grad', False, 'if True, do not use second derivatives in meta-optimization (for speed)')

## Logging, saving, and testing options
flags.DEFINE_bool('log', True, 'if false, do not log summaries, for debugging code.')
flags.DEFINE_string('logdir', '/tmp/data', 'directory for summaries and checkpoints.')
flags.DEFINE_bool('resume', True, 'resume training if there is a model available')
flags.DEFINE_bool('train', True, 'True to train, False to test.')
flags.DEFINE_integer('test_iter', -1, 'iteration to load model (-1 for latest model)')
flags.DEFINE_bool('test_set', False, 'Set to true to test on the the test set, False for the validation set.')
flags.DEFINE_integer('train_update_batch_size', -1, 'number of examples used for gradient update during training (use if you want to test with a different number).')
flags.DEFINE_float('train_update_lr', -1, 'value of inner gradient step step during training. (use if you want to test with a different value)') # 0.1 for omniglot

graphProgress = True

np.random.seed(124202)
random.seed(120293442)

def train(model, saver, sess, exp_string, data_generator, resume_itr=0):
    SUMMARY_INTERVAL = 50
    SAVE_INTERVAL = 50
    GRAPH_INTERVAL = 99
    if FLAGS.datasource == 'sinusoid':
        PRINT_INTERVAL = 10
        TEST_PRINT_INTERVAL = PRINT_INTERVAL*5
    else:
        PRINT_INTERVAL = 100
        TEST_PRINT_INTERVAL = PRINT_INTERVAL*5
    if FLAGS.log:
        train_writer = tf.summary.FileWriter(FLAGS.logdir + '/' + exp_string, sess.graph)
    print('Done initializing, starting training.')
    prelosses, postlosses, auto_losses_list = [], [], []
    num_classes = data_generator.num_classes # for classification, 1 otherwise
    multitask_weights, reg_weights = [], []
    val_filename = FLAGS.logdir +'/'+ exp_string + '/' + 'validation_loss.csv'
    val_file = open(val_filename,'w')
    train_filename = FLAGS.logdir +'/'+ exp_string + '/' + 'train_loss.csv'
    train_file = open(train_filename,'w')
    auto_losses_list = []
    

    for itr in range(resume_itr, FLAGS.pretrain_iterations + FLAGS.metatrain_iterations+ FLAGS.encoder_iterations):
        feed_dict = {}
        if 'generate' in dir(data_generator):
            batch_x, batch_y, amp, phase = data_generator.generate()
            inputa = batch_x[:, :num_classes*FLAGS.update_batch_size, :]
            ina1 = inputa[:,:,0,:,:]
            ina2 = inputa[:,:,1,:,:]
            ina3 = inputa[:,:,2,:,:]
            #print(ina1.shape)
            #print(ina1[0][0].shape)
            x0,y0 = get_xx_yy(ina1[0][0])
            #print("CoM Test 1: " , x0, y0)
            x0,y0 = get_xx_yy(ina2[0][0])
            #print("CoM Test 2: " , x0, y0)
            x0,y0 = get_xx_yy(ina3[0][0])
            #print("CoM Test3: " , x0, y0)
            #print("The first ina is: " , ina1)
            #print("Input a: " , inputa.shape)
            #print(inputa[0][0][0])
            #print("Label a: " , labela.shape)
            inputb = batch_x[:, num_classes*FLAGS.update_batch_size:, :] # b used for testing
            inb1 = inputb[:,:,0,:,:]
            inb2 = inputb[:,:,1,:,:]
            inb3 = inputb[:,:,2,:,:]
            labela = batch_y[:, :num_classes*FLAGS.update_batch_size, :]
            laba = labela[:,:,0,:,:]
            labelb = batch_y[:, num_classes*FLAGS.update_batch_size:, :]
            labb = labelb[:,:,0,:,:]
            #print("Ina1 Shape: " , ina1.shape)
            #print("Laa Shape: " , laba.shape)
            #print("Lab Shape: " , labb.shape)
            #os.exit()
            #print("InputB: " , inputa)
            #print("LabelB: " , labela)
            #os.exit()


            #feed_dict = {model.inputa: inputa, model.inputb: inputb,  model.labela: labela, model.labelb: labelb}
            feed_dict = {model.inputa1:ina1,model.inputa2:ina2,model.inputa3:ina3,model.inputb1:inb1,model.inputb2:inb2,model.inputb3:inb3,model.labela:laba,model.labelb:labb}
        action = ""

        if itr < FLAGS.encoder_iterations:
            input_tensors = [model.autotrain_op]
            action = "Auto-regressive"
            #print("Auto-regressive training......")
        elif (itr-FLAGS.encoder_iterations) < FLAGS.pretrain_iterations:
            input_tensors = [model.pretrain_op]
            #print("Pretraining the network......")
            action = "pretrain"

        else:
            input_tensors = [model.metatrain_op]
            #print("Metatraining the network......")
            action = "metatrain"

        if (itr % SUMMARY_INTERVAL == 0 or itr % PRINT_INTERVAL == 0):
            input_tensors.extend([model.summ_op, model.total_loss1, model.total_losses2[FLAGS.num_updates-1],model.auto_losses])
            # lossa, lossb, autoloss
            

        result = sess.run(input_tensors, feed_dict)
        

        if (itr % SUMMARY_INTERVAL == 0 or itr % PRINT_INTERVAL == 0):
            auto_losses_list.append(result[-1])
            prelosses.append(result[-3]) #  Loss a
            postlosses.append(result[-2]) # Loss b
        
        #auto_losses_list.append(result[-1])
        #prelosses.append(result[-1])
        #postlosses.append(result[-1])
        #print("Auto losses list: " , auto_losses_list)

        if graphProgress and (itr % GRAPH_INTERVAL) == 0:
            print_result = sess.run([model.auto_out_a,model.auto_out_b,model.outputbs[FLAGS.num_updates-1]],feed_dict)

            auto_output = print_result[2].reshape(FLAGS.meta_batch_size,100,39,39)
            correct_out = labb
            #print(auto_output.shape)
            #print(correct_out.shape)

            val_one = auto_output[0][0]
            x0,y0 = get_xx_yy(val_one)
            print("Graphing data.....")
            print("   Prediction CoM Auto from NN: " , x0, y0)
            val_two = correct_out[0][0]
            x0,y0 = get_xx_yy(val_two)
            print("   Prediction CoM Auto true Re: " , x0, y0)
            plt.switch_backend('agg')
            fig=plt.figure(figsize=(1, 3))
            fig.add_subplot(1, 3, 1)
            plt.imshow(val_one)
            fig.add_subplot(1, 3, 2)
            plt.imshow(val_two)
            fig.add_subplot(1, 3, 3)
            plt.imshow(val_one-val_two)
            myId = random.randint(0,1000)
            fig.savefig(FLAGS.logdir + '/' + exp_string+"/pred_encode_"+str(myId)+".pdf", bbox_inches='tight')
            plt.close()

            #print(print_result)
            auto_out_a = print_result[0].reshape(FLAGS.meta_batch_size,100,39,39)
            correct_out_a = ina1
            #First ele
            val_one = auto_out_a[0][0]
            x0,y0 = get_xx_yy(val_one)
            print("   Auto-Enc CoM Auto from NN: " , x0, y0)
            val_two = correct_out_a[0][0]
            x0,y0 = get_xx_yy(val_two)
            print("   Auto-Enc CoM Auto true Re: " , x0, y0)
            plt.switch_backend('agg')
            fig=plt.figure(figsize=(1, 3))
            fig.add_subplot(1, 3, 1)
            plt.imshow(val_one)
            fig.add_subplot(1, 3, 2)
            plt.imshow(val_two)
            fig.add_subplot(1, 3, 3)
            plt.imshow(val_one-val_two)
            myId = random.randint(0,1000)
            fig.savefig(FLAGS.logdir + '/' + exp_string+"/auto_encode_"+str(myId)+".pdf", bbox_inches='tight')
            plt.close()
        if itr % SUMMARY_INTERVAL == 0:
            train_writer.add_summary(result[1], itr)
 
        if (itr!=0) and itr % PRINT_INTERVAL == 0:
            print_str = str(itr) + " Action: " + str(action) + " "
            
            if itr < FLAGS.encoder_iterations:
                print_str = 'Encoder  Iteration (encoder loss):' + str(itr)
            elif (itr-FLAGS.encoder_iterations) < FLAGS.pretrain_iterations:
                print_str = 'Pretrain Iteration (propoga loss):' + str(itr-FLAGS.encoder_iterations)
            else:
                print_str = 'Iteration          (meta    loss):' + str(itr - FLAGS.encoder_iterations - FLAGS.pretrain_iterations)
            print_str = str(itr) + " : " + str(action) + " AutoLosses : " + str(np.mean(auto_losses_list)) + " Loss a : " + str(np.mean(prelosses)) + " Loss b: " + str(np.mean(postlosses)) 
            print(print_str)
            # structure is iteration, a, b, auto            
            train_file.write(str(itr) +"," + str(np.mean(prelosses)) + ', ' + str(np.mean(postlosses))+"," + str(np.mean(auto_losses_list)) + "\n")
            prelosses, postlosses, auto_losses_list = [], [], []

        if (itr!=0) and itr % SAVE_INTERVAL == 0:
            saver.save(sess, FLAGS.logdir + '/' + exp_string + '/model' + str(itr))

        # sinusoid is infinite data, so no need to test on meta-validation set.
        if (itr!=0) and itr % TEST_PRINT_INTERVAL == 0:
            if 'generate' not in dir(data_generator):
                feed_dict = {}
            
                input_tensors = [model.metaval_total_loss1, model.metaval_total_losses2[FLAGS.num_updates-1], model.summ_op]
            else:
                #print("----Testing-----")
                batch_x, batch_y, amp, phase = data_generator.generate(train=False)
                inputa = batch_x[:, :num_classes*FLAGS.update_batch_size, :]
                ina1 = inputa[:,:,0,:,:]
                ina2 = inputa[:,:,1,:,:]
                ina3 = inputa[:,:,2,:,:]

                #print(inputa[0])
                #print("Label a: " , labela.shape)
                inputb = batch_x[:, num_classes*FLAGS.update_batch_size:, :] # b used for testing
                inb1 = inputb[:,:,0,:,:]
                inb2 = inputb[:,:,1,:,:]
                inb3 = inputb[:,:,2,:,:]
                labela = batch_y[:, :num_classes*FLAGS.update_batch_size, :]
                laba = labela[:,:,0,:,:]


                labelb = batch_y[:, num_classes*FLAGS.update_batch_size:, :]
                labb = labelb[:,:,0,:,:]
                #print("InA1 Data: ", ina1.shape)
                #print(ina1)
                #print("Laba: " , laba.shape)
                #print(laba)

                #print("Input a: " , inputa.shape)
                #print("Label a: " , laba.shape)

                #print("inputa: " , inputa[0])
                #print("Ina Shape: " , inputa.shape)
                #print("Inb Shape: " , inputb.shape)
                #my = input("hi")
                feed_dict = {model.inputa1:ina1,model.inputa2:ina2,model.inputa3:ina3,model.inputb1:inb1,model.inputb2:inb2,model.inputb3:inb3,model.labela:laba,model.labelb:labb,model.meta_lr: 0.0}
                #feed_dict = {model.inputa: inputa, model.inputb: inputb,  model.labela: labela, model.labelb: labelb, model.meta_lr: 0.0}
                
                input_tensors = [model.total_loss1, model.total_losses2[FLAGS.num_updates-1],model.auto_losses,model.outputbs[FLAGS.num_updates-1]]

            result = sess.run(input_tensors, feed_dict)
            #print_reuslt = sess.run(model.result,feed_dict)
            #print("print reuslt: " , print_reuslt)
            #We need to nromalize it out. 

            pre_loss = result[0]/100.0*FLAGS.meta_batch_size
            print("meta batch size: " , FLAGS.meta_batch_size)
            post_loss = result[1]/100.0*FLAGS.meta_batch_size
            #auto_loss = result[2]/100.0*FLAGS.meta
            auto_loss = result[2]/100.0*FLAGS.meta_batch_size
            #print("Output: " , result[3])
            print('Validation results: ' + str(pre_loss) + ', ' + str(post_loss))
            print('Auto-Encoding loss: ' + str(auto_loss))
            
            val_file.write(str(itr - FLAGS.pretrain_iterations) +"," + str(pre_loss) + ', ' + str(post_loss)+", " + str(auto_loss) + "\n")
            train_file.flush()
            val_file.flush()



    saver.save(sess, FLAGS.logdir + '/' + exp_string +  '/model' + str(itr))

# calculated for omniglot
NUM_TEST_POINTS = 20

def test(model, saver, sess, exp_string, data_generator, test_num_updates=None):
    num_classes = data_generator.num_classes # for classification, 1 otherwise

    metaval_accuracies = []

    for _ in range(NUM_TEST_POINTS):
        if 'generate' not in dir(data_generator):
            feed_dict = {}
            feed_dict = {model.meta_lr : 0.0}
        else:
            batch_x, batch_y, amp, phase = data_generator.generate(train=False,numTestBatches=1)
            #print("generating...")
            #print(batch_x)

            inputa = batch_x[:, :num_classes*FLAGS.update_batch_size, :]
            ina1 = inputa[:,:,0,:,:]
            ina2 = inputa[:,:,1,:,:]
            ina3 = inputa[:,:,2,:,:]
            #print("Input a: " , inputa.shape)
            #print(inputa[0])
            #print("Label a: " , labela.shape)
            inputb = batch_x[:, num_classes*FLAGS.update_batch_size:, :] # b used for testing
            inb1 = inputb[:,:,0,:,:]
            inb2 = inputb[:,:,1,:,:]
            inb3 = inputb[:,:,2,:,:]
            labela = batch_y[:, :num_classes*FLAGS.update_batch_size, :]
            laba = labela[:,:,0,:,:]

            labelb = batch_y[:, num_classes*FLAGS.update_batch_size:, :]
            labb = labelb[:,:,0,:,:]

            #inputa = batch_x[:, :num_classes*FLAGS.update_batch_size, :]
            #print("in a: " , inputa)
            #print("Shape: " , inputa.shape)
            #inputb = batch_x[:,num_classes*FLAGS.update_batch_size:, :]
            #labela = batch_y[:, :num_classes*FLAGS.update_batch_size, :]
            #labelb = batch_y[:,num_classes*FLAGS.update_batch_size:, :]
            feed_dict = {model.inputa1:ina1,model.inputa2:ina2,model.inputa3:ina3,model.inputb1:inb1,model.inputb2:inb2,model.inputb3:inb3,model.labela:laba,model.labelb:labb, model.meta_lr: 0.0}

            #feed_dict = {model.inputa: inputa, model.inputb: inputb,  model.labela: labela, model.labelb: labelb, model.meta_lr: 0.0}

        result = sess.run([model.total_loss1] +  model.total_losses2, feed_dict)

        print_result = sess.run([model.auto_out_a,model.auto_out_b,model.outputbs[FLAGS.num_updates-1]],feed_dict)
        print("Output: " , model.outputbs[FLAGS.num_updates-1])
        #print(result)
        metaval_accuracies.append(result)

    metaval_accuracies = np.array(metaval_accuracies)/100.0 #Because we fixed batch size to be 100
    means = np.mean(metaval_accuracies, 0)
    stds = np.std(metaval_accuracies, 0)
    ci95 = 1.96*stds/np.sqrt(NUM_TEST_POINTS)

    print('Mean validation accuracy/loss, stddev, and confidence intervals')
    print((means, stds, ci95))

    out_filename = FLAGS.logdir +'/'+ exp_string + '/' + 'test_ubs' + str(FLAGS.update_batch_size) + '_stepsize' + str(FLAGS.update_lr) + '.csv'
    out_pkl = FLAGS.logdir +'/'+ exp_string + '/' + 'test_ubs' + str(FLAGS.update_batch_size) + '_stepsize' + str(FLAGS.update_lr) + '.pkl'
    with open(out_pkl, 'wb') as f:
        pickle.dump({'mses': metaval_accuracies}, f)
    with open(out_filename, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['update'+str(i) for i in range(len(means))])
        writer.writerow(means)
        writer.writerow(stds)
        writer.writerow(ci95)

def main():
    if FLAGS.datasource == 'sinusoid':
        if FLAGS.train:
            test_num_updates = 5
        else:
            test_num_updates = 2
    else:
        if FLAGS.datasource == 'miniimagenet':
            if FLAGS.train == True:
                test_num_updates = 1  # eval on at least one update during training
            else:
                test_num_updates = 10
        else:
            test_num_updates = 10

    if FLAGS.train == False:
        orig_meta_batch_size = FLAGS.meta_batch_size
        # always use meta batch size of 1 when testing.
        FLAGS.meta_batch_size = 1

    if FLAGS.datasource == 'sinusoid':
        data_generator = DataGenerator(FLAGS.update_batch_size*2, FLAGS.meta_batch_size)
    else:
        if FLAGS.metatrain_iterations == 0 and FLAGS.datasource == 'miniimagenet':
            assert FLAGS.meta_batch_size == 1
            assert FLAGS.update_batch_size == 1
            data_generator = DataGenerator(1, FLAGS.meta_batch_size)  # only use one datapoint,
        else:
            if FLAGS.datasource == 'miniimagenet': # TODO - use 15 val examples for imagenet?
                if FLAGS.train:
                    data_generator = DataGenerator(FLAGS.update_batch_size+15, FLAGS.meta_batch_size)  # only use one datapoint for testing to save memory
                else:
                    data_generator = DataGenerator(FLAGS.update_batch_size*2, FLAGS.meta_batch_size)  # only use one datapoint for testing to save memory
            else:
                data_generator = DataGenerator(FLAGS.update_batch_size*2, FLAGS.meta_batch_size)  # only use one datapoint for testing to save memory


    dim_output = data_generator.dim_output
    if FLAGS.baseline == 'oracle':
        assert FLAGS.datasource == 'sinusoid'
        dim_input = 3
        FLAGS.pretrain_iterations += FLAGS.metatrain_iterations
        FLAGS.metatrain_iterations = 0
    else:
        dim_input = data_generator.dim_input

    if FLAGS.datasource == 'miniimagenet' or FLAGS.datasource == 'omniglot':
        tf_data_load = True
        num_classes = data_generator.num_classes

        if FLAGS.train: # only construct training model if needed
            random.seed(5)
            image_tensor, label_tensor = data_generator.make_data_tensor()
            inputa = tf.slice(image_tensor, [0,0,0], [-1,num_classes*FLAGS.update_batch_size, -1])
            inputb = tf.slice(image_tensor, [0,num_classes*FLAGS.update_batch_size, 0], [-1,-1,-1])
            labela = tf.slice(label_tensor, [0,0,0], [-1,num_classes*FLAGS.update_batch_size, -1])
            labelb = tf.slice(label_tensor, [0,num_classes*FLAGS.update_batch_size, 0], [-1,-1,-1])
            input_tensors = {'inputa': inputa, 'inputb': inputb, 'labela': labela, 'labelb': labelb}

        random.seed(6)
        image_tensor, label_tensor = data_generator.make_data_tensor(train=False)
        inputa = tf.slice(image_tensor, [0,0,0], [-1,num_classes*FLAGS.update_batch_size, -1])
        inputb = tf.slice(image_tensor, [0,num_classes*FLAGS.update_batch_size, 0], [-1,-1,-1])
        labela = tf.slice(label_tensor, [0,0,0], [-1,num_classes*FLAGS.update_batch_size, -1])
        labelb = tf.slice(label_tensor, [0,num_classes*FLAGS.update_batch_size, 0], [-1,-1,-1])
        metaval_input_tensors = {'inputa': inputa, 'inputb': inputb, 'labela': labela, 'labelb': labelb}
    else:
        tf_data_load = False
        input_tensors = None

    model = MAML(dim_input, dim_output, test_num_updates=test_num_updates)
    if FLAGS.train or not tf_data_load:
        model.construct_model(input_tensors=input_tensors, prefix='metatrain_')
    if tf_data_load:
        model.construct_model(input_tensors=metaval_input_tensors, prefix='metaval_')
    model.summ_op = tf.summary.merge_all()

    saver = loader = tf.train.Saver(max_to_keep=10)

    #saver = loader = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), max_to_keep=10)

    sess = tf.InteractiveSession()

    if FLAGS.train == False:
        # change to original meta batch size when loading model.
        FLAGS.meta_batch_size = orig_meta_batch_size

    if FLAGS.train_update_batch_size == -1:
        FLAGS.train_update_batch_size = FLAGS.update_batch_size
    if FLAGS.train_update_lr == -1:
        FLAGS.train_update_lr = FLAGS.update_lr

    exp_string = 'cls_'+str(FLAGS.num_classes)+'.mbs_'+str(FLAGS.meta_batch_size) + '.ubs_' + str(FLAGS.train_update_batch_size) + '.numstep' + str(FLAGS.num_updates) + '.updatelr' + str(FLAGS.train_update_lr)

    if FLAGS.num_filters != 64:
        exp_string += 'hidden' + str(FLAGS.num_filters)
    if FLAGS.max_pool:
        exp_string += 'maxpool'
    if FLAGS.stop_grad:
        exp_string += 'stopgrad'
    if FLAGS.baseline:
        exp_string += FLAGS.baseline
    if FLAGS.norm == 'batch_norm':
        exp_string += 'batchnorm'
    elif FLAGS.norm == 'layer_norm':
        exp_string += 'layernorm'
    elif FLAGS.norm == 'None':
        exp_string += 'nonorm'
    else:
        print('Norm setting not recognized.')

    resume_itr = 0
    model_file = None

    tf.global_variables_initializer().run()
    tf.train.start_queue_runners()

    if FLAGS.resume or not FLAGS.train:
        print("Seeing if resume....")
        print("File string: " , FLAGS.logdir + '/' + exp_string)
        model_file = tf.train.latest_checkpoint(FLAGS.logdir + '/' + exp_string)
        print("model file name: ", model_file)
        if FLAGS.test_iter > 0:
            model_file = model_file[:model_file.index('model')] + 'model' + str(FLAGS.test_iter)
        if model_file:
            ind1 = model_file.index('model')
            resume_itr = int(model_file[ind1+5:])
            print("Restoring model weights from " + model_file)
            saver.restore(sess, model_file)

    if FLAGS.train:
        train(model, saver, sess, exp_string, data_generator, resume_itr)
    else:
        test(model, saver, sess, exp_string, data_generator, test_num_updates)

# This is for bouncing ball, to visualize the errors.


import matplotlib.pylab as plt
def graphPoints(inval,lab,true):
    print("Graping....")
    for i in xrange(0,len(inval)):
        in_val_x = inval[i][0:5:2]
        in_val_y = inval[i][1:6:2]
        la_val = lab[i]
        tr_val = true[i]
        outX = tr_val[0]
        outY = tr_val[1]

        print("x vals", list(in_val_x),list([la_val[0]]))
        #print("In Val: " , in_val)
        #print("LA : " , la_val)
        #print("TR : " , tr_val)
        col = np.random.rand(3,)

        plt.plot(list(in_val_x) + list([la_val[0]]), list(in_val_y) + list([la_val[1]]), '-o',c=col)
        plt.plot(outX,outY,'-o',c=col)
        plt.show()
        #os.exit()
    plt.show()

    os.exit() 
        
    pass
def get_xx_yy(A):
    A = (A >= 0.5)*1.0*0.025
    w = np.arange(0,A.shape[0])
    val1 = (w*np.sum(A,0)).sum()
    val2 = (w*np.sum(A,1)).sum()
    weight = (np.sum(A,0).sum(0)+0.000000001)
    x = (val1 + 1e-8)/(weight + 1e-8)
    y = (val2 + 1e-8)/(weight + 1e-8)
    return x, y

if __name__ == "__main__":
    main()
