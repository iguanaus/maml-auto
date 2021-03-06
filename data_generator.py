""" Code for loading data. """
import numpy as np
import os
import random
import tensorflow as tf

from tensorflow.python.platform import flags
from utils import get_images
import pickle

import matplotlib.pylab as plt


FLAGS = flags.FLAGS
#random.seed(123490234)

#This method will create all the data.
#task_id = "C-sin"
# task_id = "C-sin"
# task_id = "bounce-states"
#num_shots = 10
#dataset_PATH = "data/"
#filename = dataset_PATH + task_id + "_{0}-shot_2.p".format(num_shots)
#tasks = pickle.load(open(filename, "rb"))
filename = "data/bounce-images_100-shot-2.p"
#filename = "data/bounce-states_100-shot_2.p"
#filename = "data/C-sin_10-shot_legit_stateform.p"
#batch_size = 25

tasks = pickle.load(open(filename, "rb"))

def convertDataImage_Real(batch_size,myTrain,shouldPlot=False):
    num_batches = len(myTrain)/batch_size
    print("Packaging data.......")
    print("Num batches: " , num_batches)
    allTrainData = []
    # For each batch. 
    for i in xrange(0,num_batches):
        # Figure out the corresponding tasks. 
        tasks_for_batch = myTrain[i*batch_size:(i+1)*batch_size]
        inputAll = np.array([])
        labelAll = np.array([])
        myLastEles = []
        # For each individual task. 
        # There should be 100 examples, of 3 screens, with 39 x 39. 
        for task in tasks_for_batch:
            data = task[0]
            #print("Data for this task : " , data)
            info = task[1]
            boxCords = info['z'].reshape(-1,2)*.025
            xCords = list(boxCords[:,0])
            yCords = list(boxCords[:,1])
            #Add last element to it.
            xCords.append(xCords[0])
            yCords.append(yCords[0])
            if shouldPlot:
                plt.plot(xCords,yCords)
            #print("My data: " , data[0][0].shape)
            inputa = data[0][0].reshape(-1,3,39,39)
            # This should by a set of 
            print(inputa.shape)
            x0,y0 = get_xx_yy(inputa[0][0])
            print("CoM: " , x0, y0)

            #print(inputa[0][0])


            #inputa = data[0][0].reshape(-1,3,2,1)
            #inputa_fin = np.tile(inputa,(1,1,21,42))
            #inputa = data[0][0].reshape(-1,6) # This is doing exactly what we want
            onlyNextLabela = data[0][1][:,0:1,:].reshape(-1,1,39,39)
            #print("New label a: " , onlyNextLabela.shape)
            #onlyNextLabela_fin = np.tile(onlyNextLabela,(1,1,21,42))

            inputb = data[1][0].reshape(-1,3,39,39)
            finalVals0 = data[1][1][:,:,:].reshape(-1,39,39)

            #inputb_fin = np.tile(inputb,(1,1,21,42))
            #inputa = data[0][0].reshape(-1,6) # This is doing exactly what we want
            myLastEles.append(finalVals0)
            onlyNextLabelb = data[1][1][:,0:1,:].reshape(-1,1,39,39)
            #onlyNextLabelb_fin = np.tile(onlyNextLabelb,(1,1,21,42))

            #.reshape(-1,2)

            #inputb = data[1][0].reshape(-1,6)
            #onlyNextLabelb = data[1][1][:,0:1,:].reshape(-1,2) #This was pulling from the same set. 
            inputs = np.vstack((inputa,inputb))
            #print(inputs.shape)
            inputs = inputs.reshape(1,-1,3,39,39)
            #print("Final inputs shape: " , inputs.shape)

            #inputs = inputs.reshape(1,-1,6)

            labels = np.vstack((onlyNextLabela,onlyNextLabelb))
            #print("labels shape: " , labels.shape)
            labels = labels.reshape(1,-1,1,39,39)
            #print("Inputs shape: ",inputs)
            #print("Labels shape: " , labels)

            if shouldPlot:
                for j in xrange(0, 200):
                    taskX = inputs[0][j][0:5:2] 
                    taskY = inputs[0][j][1:6:2]
                    outX = labels[0][j][0]
                    outY = labels[0][j][1]
                    pltX = taskX + outX
                    #print(pltX)
                    #print(outX)
                    plt.plot(list(taskX) + list([outX]), list(taskY) + [outY], '-o')
            if inputAll.size == 0:
                inputAll = inputs
                labelAll = labels
            else:
                inputAll = np.vstack((inputAll,inputs))
                labelAll = np.vstack((labelAll,labels))
            if shouldPlot:
                plt.show()
            #break
        allTrainData.append([inputAll,labelAll,myLastEles,0])

    #b_x,b_y,amp,phase = allTrainData[1]
    #inputb = b_x[:,:FLAGS.update_batch_size]
    #labelb = b_y[:,:FLAGS.update_batch_size]
    print("Done packaging data.....")

    return allTrainData


class DataGenerator(object):
    """
    Data Generator capable of generating batches of sinusoid or Omniglot data.
    A "class" is considered a class of omniglot digits or a particular sinusoid function.
    """
    def __init__(self, num_samples_per_class, batch_size, config={}):
        """
        Args:
            num_samples_per_class: num samples to generate per class in one batch
            batch_size: size of meta batch size (e.g. number of functions)
        """
        self.allTrainData = None
        self.allTestData = None
        self.iterCount = 0

        self.batch_size = batch_size
        self.num_samples_per_class = num_samples_per_class
        self.num_classes = 1  # by default 1 (only relevant for classification problems)

        if FLAGS.datasource == 'sinusoid':
            self.generate = self.generate_sinusoid_batch
            self.amp_range = config.get('amp_range', [0.1, 5.0])
            self.phase_range = config.get('phase_range', [0, np.pi])
            self.input_range = config.get('input_range', [-5.0, 5.0])
            self.dim_input = 6
            self.dim_output = 2
        elif 'omniglot' in FLAGS.datasource:
            self.num_classes = config.get('num_classes', FLAGS.num_classes)
            self.img_size = config.get('img_size', (28, 28))
            self.dim_input = np.prod(self.img_size)
            self.dim_output = self.num_classes
            # data that is pre-resized using PIL with lanczos filter
            data_folder = config.get('data_folder', './data/omniglot_resized')

            character_folders = [os.path.join(data_folder, family, character) \
                for family in os.listdir(data_folder) \
                if os.path.isdir(os.path.join(data_folder, family)) \
                for character in os.listdir(os.path.join(data_folder, family))]
            random.seed(1)
            random.shuffle(character_folders)
            num_val = 100
            num_train = config.get('num_train', 1200) - num_val
            self.metatrain_character_folders = character_folders[:num_train]
            if FLAGS.test_set:
                self.metaval_character_folders = character_folders[num_train+num_val:]
            else:
                self.metaval_character_folders = character_folders[num_train:num_train+num_val]
            self.rotations = config.get('rotations', [0, 90, 180, 270])
        elif FLAGS.datasource == 'miniimagenet':
            self.num_classes = config.get('num_classes', FLAGS.num_classes)
            self.img_size = config.get('img_size', (84, 84))
            self.dim_input = np.prod(self.img_size)*3
            self.dim_output = self.num_classes
            metatrain_folder = config.get('metatrain_folder', './data/miniImagenet/train')
            if FLAGS.test_set:
                metaval_folder = config.get('metaval_folder', './data/miniImagenet/test')
            else:
                metaval_folder = config.get('metaval_folder', './data/miniImagenet/val')

            metatrain_folders = [os.path.join(metatrain_folder, label) \
                for label in os.listdir(metatrain_folder) \
                if os.path.isdir(os.path.join(metatrain_folder, label)) \
                ]
            metaval_folders = [os.path.join(metaval_folder, label) \
                for label in os.listdir(metaval_folder) \
                if os.path.isdir(os.path.join(metaval_folder, label)) \
                ]
            self.metatrain_character_folders = metatrain_folders
            self.metaval_character_folders = metaval_folders
            self.rotations = config.get('rotations', [0])
        else:
            raise ValueError('Unrecognized data source')




    def setupData(self,num_tasks=100,numTest=100,numTestBatches=1):
        oldVersion = False
        if oldVersion:
            print("setupData. setting up data.....")
            self.allTrainData = []
            self.allTestData = []
            #This is how many unique task to make.
            for i in xrange(0,num_tasks):
                self.allTrainData.append(self.generate_sinusoid_batch(usePreValues=False))
            ordd = self.batch_size
            self.batch_size = numTest
            for i in xrange(0,numTestBatches):
                self.allTestData.append(self.generate_sinusoid_batch(usePreValues=False))
            self.batch_size = ordd
            print("setupData. Done setting up data....")
        else:
            self.allTrainData = convertDataImage_Real(self.batch_size,tasks['tasks_train'])
            self.allTestData = convertDataImage_Real(numTestBatches,tasks['tasks_test'])
            #print("All train: " , self.allTrainData)
            #print("All tests: " , self.allTestData)
            print(len(self.allTrainData))
            print(len(self.allTestData))

        print("Done with setup....")

        #print("sell all tasks: " , self.allTrainData)
        #Then you cPan just call this
        #generate_sinusoid_batch
    def getPreData(self,num_tasks=100,train=True,numTestBatches=1):
        #random.seed(123490234)
        
        #print("num tasks: " , num_tasks)
        #print("Rand id: " , ranId)
        if train:
            idRet = self.iterCount
            self.iterCount += 1
            if (self.iterCount > (len(self.allTrainData)-1)):
                self.iterCount = 0
            #print("Id return: " , idRet)
            return self.allTrainData[idRet]
        else:
            if numTestBatches > 1:
                numTestBatches = len(self.allTestData)
            #print("Test data val: ")
            print(len(self.allTestData))
            #idRet = self.iterCount
            #self.iterCount += 1
            #if (self.iterCount > (numTestBatches-1)):
            #    self.iterCount = 0
            idRet = 0
            #print("testing..: " , idRet)
            #print("Dim: " , len(self.allTestData[idRet]))
            return self.allTestData[idRet]

    def make_data_tensor(self, train=True):
        if train:
            folders = self.metatrain_character_folders
            # number of tasks, not number of meta-iterations. (divide by metabatch size to measure)
            num_total_batches = 200000
        else:
            folders = self.metaval_character_folders
            num_total_batches = 600

        # make list of files
        print('Generating filenames')
        all_filenames = []
        for _ in range(num_total_batches):
            sampled_character_folders = random.sample(folders, self.num_classes)
            random.shuffle(sampled_character_folders)
            labels_and_images = get_images(sampled_character_folders, range(self.num_classes), nb_samples=self.num_samples_per_class, shuffle=False)
            # make sure the above isn't randomized order
            labels = [li[0] for li in labels_and_images]
            filenames = [li[1] for li in labels_and_images]
            all_filenames.extend(filenames)

        # make queue for tensorflow to read from
        filename_queue = tf.train.string_input_producer(tf.convert_to_tensor(all_filenames), shuffle=False)
        print('Generating image processing ops')
        image_reader = tf.WholeFileReader()
        _, image_file = image_reader.read(filename_queue)
        if FLAGS.datasource == 'miniimagenet':
            image = tf.image.decode_jpeg(image_file, channels=3)
            image.set_shape((self.img_size[0],self.img_size[1],3))
            image = tf.reshape(image, [self.dim_input])
            image = tf.cast(image, tf.float32) / 255.0
        else:
            image = tf.image.decode_png(image_file)
            image.set_shape((self.img_size[0],self.img_size[1],1))
            image = tf.reshape(image, [self.dim_input])
            image = tf.cast(image, tf.float32) / 255.0
            image = 1.0 - image  # invert
        num_preprocess_threads = 1 # TODO - enable this to be set to >1
        min_queue_examples = 256
        examples_per_batch = self.num_classes * self.num_samples_per_class
        batch_image_size = self.batch_size  * examples_per_batch
        print('Batching images')
        images = tf.train.batch(
                [image],
                batch_size = batch_image_size,
                num_threads=num_preprocess_threads,
                capacity=min_queue_examples + 3 * batch_image_size,
                )
        all_image_batches, all_label_batches = [], []
        print('Manipulating image data to be right shape')
        for i in range(self.batch_size):
            image_batch = images[i*examples_per_batch:(i+1)*examples_per_batch]

            if FLAGS.datasource == 'omniglot':
                # omniglot augments the dataset by rotating digits to create new classes
                # get rotation per class (e.g. 0,1,2,0,0 if there are 5 classes)
                rotations = tf.multinomial(tf.log([[1., 1.,1.,1.]]), self.num_classes)
            label_batch = tf.convert_to_tensor(labels)
            new_list, new_label_list = [], []
            for k in range(self.num_samples_per_class):
                class_idxs = tf.range(0, self.num_classes)
                class_idxs = tf.random_shuffle(class_idxs)

                true_idxs = class_idxs*self.num_samples_per_class + k
                new_list.append(tf.gather(image_batch,true_idxs))
                
                new_label_list.append(tf.gather(label_batch, true_idxs))
            new_list = tf.concat(new_list, 0)  # has shape [self.num_classes*self.num_samples_per_class, self.dim_input]
            new_label_list = tf.concat(new_label_list, 0)
            all_image_batches.append(new_list)
            all_label_batches.append(new_label_list)
        all_image_batches = tf.stack(all_image_batches)
        all_label_batches = tf.stack(all_label_batches)
        all_label_batches = tf.one_hot(all_label_batches, self.num_classes)
        return all_image_batches, all_label_batches

    def generate_sinusoid_batch(self, train=True, input_idx=None,usePreValues=True,numTotal=None,numTestBatches=100):
        #if FLAGS.train==True:
        if numTotal == None:
            numTotal = FLAGS.limit_task_num

        if FLAGS.limit_task == True and usePreValues:
            #print("us")
            if self.allTrainData == None:
                self.setupData(num_tasks=numTotal,numTestBatches=numTestBatches)
            return self.getPreData(num_tasks=numTotal,train=train,numTestBatches=numTestBatches)

def get_xx_yy(A):
    A = (A >= 0.5)*1.0*0.025
    w = np.arange(0,A.shape[0])
    val1 = (w*np.sum(A,0)).sum()
    val2 = (w*np.sum(A,1)).sum()
    weight = (np.sum(A,0).sum(0)+0.000000001)
    x = (val1 + 1e-8)/(weight + 1e-8)
    y = (val2 + 1e-8)/(weight + 1e-8)
    return x, y
