#!/usr/bin/env/ python
# ECBM E4040 Fall 2020 Assignment 2
# TensorFlow custom CNN model
import tensorflow as tf
from utils.neuralnets.cnn.model_LeNet import LeNet
from utils.image_generator import ImageGenerator
from tqdm import tqdm
import datetime


class MyLeNet_trainer():
    """
    X_train: Train Images. It should be a 4D array like (n_train_images, img_len, img_len, channel_num).
    y_train: Train Labels. It should be a 1D vector like (n_train_images, )
    X_val: Validation Images. It should be a 4D array like (n_val_images, img_len, img_len, channel_num).
    y_val: Validation Labels. It should be a 1D vector like (n_val_images, )
    epochs: Number of training epochs
    batch_size: batch_size while training
    lr: learning rate of optimizer
    """
    def __init__(self,X_train, y_train, X_val, y_val, epochs=10, batch_size=256, lr=1e-3):
        self.X_train = X_train.astype("float32")
        self.y_train = y_train.astype("float32")
        self.X_val = X_val.astype("float32")
        self.y_val = y_val.astype("float32")
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr

    # Initialize MyLenet model
    def init_model(self):
        self.model = LeNet(self.X_train[0].shape)

    #initialize loss function and metrics to track over training
    def init_loss(self):
        self.loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    # Initialize optimizer
    def init_optimizer(self):
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)

    # Prepare batches of train data using ImageGenerator
    def batch_train_data(self, batch_size=256, shuffle=True):
        train_data = ImageGenerator(self.X_train, self.y_train)
        #############################################################
        # TODO: create augumented data with your proposed data augmentations
        #  from part 3
        #############################################################
        
        # translated(1,1) 2.noised(0.1,3) 3.brightness(1.5) 4.flip('v')
        train_data.translate(1,1)
        train_data.noised(0.1,3)
        train_data.brightness(1.5)
        train_data.flip('v')
        train_data.create_aug_data()
        
        #############################################################
        # END TODO
        #############################################################
        
        self.train_data_next_batch = train_data.next_batch_gen(batch_size,shuffle=shuffle)
        self.n_batches = train_data.N_aug // batch_size
    
    # Define training step
    def train_step(self, images, labels, training=True):
        with tf.GradientTape() as tape:
        # training=True is always recommended as there are few layers with different
        # behavior during training versus inference (e.g. Dropout).
            predictions = self.model(images, training=training)
            loss = self.loss_function(labels, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(labels, predictions)

    # Define testing step
    def test_step(self, images, labels, training=False):
        predictions = self.model(images, training=training)
        t_loss = self.loss_function(labels, predictions)

        self.test_loss(t_loss)
        self.test_accuracy(labels, predictions)

    # train epoch
    def train_epoch(self, epoch):
        self.train_loss.reset_states()
        self.train_accuracy.reset_states()
        self.test_loss.reset_states()
        self.test_accuracy.reset_states()
        #############################################################
        # TODO: use data from ImageGenerator to train the network
        # hint: use python next feature "next(self.train_data_next_batch)""
        #############################################################
        
        # The code in example LeNet_trainer make a uncompromised situation. We need to use train_ds firstly define 
        # the next batch in an epoch. After defining it, we can choose to print(train_ds)
        # num_batch = self.X_train.shape[0]//self.batch_size
        # for i in range(num_batch):
        # batch_to_train = next(self.train_data_next_batch)
        # self.train_step(batch_to_train[0], batch_to_train[1],training=True)
        
        for batch in range (self.n_batches):
            train_x, train_y = next(self.train_data_next_batch)
            self.train_step(train_x, train_y,training=True)
            # Early stop method to avoid overfitting
#             if self.test_accuracy.result() > 90:
#                  break
#         use code in sample Lenet_trainer
#         if epoch != 0:
#             self.current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#             self.train_log_dir = 'logs/gradient_tape/' + self.current_time + '/train'
#             self.test_log_dir = 'logs/gradient_tape/' + self.current_time + '/test'
#             self.train_summary_writer = tf.summary.create_file_writer(self.train_log_dir)
#             self.test_summary_writer = tf.summary.create_file_writer(self.test_log_dir)
            
#         with self.train_summary_writer.as_default():
#             tf.summary.scalar('loss', self.train_loss.result(), step=epoch)
#             tf.summary.scalar('accuracy', self.train_accuracy.result(), step=epoch)
            
        #############################################################
        # END TODO
        #############################################################

        self.test_step(self.X_val, self.y_val)

        template = 'Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        print(template.format(self.train_loss.result(), self.train_accuracy.result() * 100, self.test_loss.result(),
                            self.test_accuracy.result() * 100))
            
    # start training
    def run(self):
        self.init_model()
        self.init_loss()
        self.init_optimizer()
        self.batch_train_data()

        for epoch in range(self.epochs):
            print('Training Epoch {}'.format(epoch + 1))
            self.train_epoch(epoch)