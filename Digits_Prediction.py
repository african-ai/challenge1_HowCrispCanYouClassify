
# coding: utf-8

# In[93]:


import tensorflow as tf
import numpy as np
import pandas as pd
import sklearn.utils as utils
from sklearn.model_selection import train_test_split
from random import shuffle
import itertools
from sklearn.model_selection import KFold




class SimpleNeuralNetwork(object):
    
    def __init__(self, param_dict, num_epochs):
        self.build_graph(param_dict)
        self.num_epochs = num_epochs
     

    def prep_data(self, train_file, test_file):
        data = pd.read_csv(train_file, index_col = 0)
        labels = data['labels']
        labels = pd.get_dummies(labels)
        data = data.drop(['labels'], axis = 1)
        
        
        testx = pd.read_csv(test_file, index_col = 0)
        
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(data, labels)
        self.X_test = testx
        return data, labels
    
    def prep_3fold(self, train_file):
        data = pd.read_csv(train_file, index_col = 0)
        labels = data['labels']
        labels = pd.get_dummies(labels)
        data = data.drop(['labels'], axis = 1)



    def batch_generator(self, data, labels):

        data = data.sample(frac=1)
        labels = labels.reindex(data.index)
        prev = 0
        for i in range(self.batch_size, len(data), self.batch_size):
            yield (data[prev:i], labels[prev:i])
            prev = i


    def get_batch(self, data, labels):

        data = data.sample(frac=1)
        labels = labels.reindex(data.index)
        return(data[0:self.batch_size], labels[0:self.batch_size])
    
    def get_test_batch(self, data):
        data = data.sample(frac=1)
        prev = 0
        for i in range(self.batch_size, len(data), self.batch_size):
            yield data[prev:i]
            prev = i
        yield pd.concat([data[prev:], data[:self.batch_size-len(data[i:])]], axis = 0)



    def build_graph(self, param_dict):
        self.batch_size = param_dict['batch_size']
        lr = param_dict['lr']
        hidden_size = param_dict['hidden_size']
        activation = param_dict['activation']
        initializer = param_dict['initializer']
        scale = param_dict['scale']

        #graph = tf.Graph()
        
            
        with tf.name_scope('placeholders'):
            self.labels = tf.placeholder(tf.float32, shape = (self.batch_size, 10))
            self.features = tf.placeholder(tf.float32, shape = (self.batch_size, 64))

        with tf.name_scope('dense_layers'):
            fc_layer_1  = tf.layers.dense(self.features, hidden_size, activation = activation, kernel_initializer = initializer, kernel_regularizer = 
tf.contrib.layers.l2_regularizer(scale))

        with tf.name_scope('loss'):

            logits =  tf.layers.dense(fc_layer_1, 10, activation = activation)
            self.pred = tf.nn.softmax(logits)
            self.loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,labels=self.labels))
            #loss = tf.reduce_sum(ce)

        with tf.name_scope('optim'):
            self.optimizer = tf.train.AdamOptimizer(lr).minimize(self.loss)


        with tf.name_scope('accuracy'):
            correct_pred = tf.equal(tf.argmax(self.pred, 1) ,tf.argmax(self.labels, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))



    def run(self, session):
        #session.as_default()
        session.run(tf.global_variables_initializer())

        cur_epoch = 0
        avg_loss = 0
        num_epochs = self.num_epochs
        acc_test = 0
        acc_train = 0
        self.max_acc = 0
        worse = 0
        step = 0

        while cur_epoch < num_epochs or worse < 50:

            for batch in self.batch_generator(self.X_train, self.y_train):
                
                ft_batch, lb_batch = batch         
                feed_dict = {self.features : ft_batch, self.labels : lb_batch}
                _, curloss, atrain = sess.run([self.optimizer, self.loss, self.accuracy], 
                                                 feed_dict = feed_dict)

                acc_train += atrain

                fv, lv = self.get_batch(self.X_val, self.y_val)
                val_dict = {self.features: fv, self.labels:lv}
                atest = sess.run(self.accuracy, feed_dict = val_dict)
                acc_test += atest


                avg_loss += curloss
                step += 1

            self.average_loss = avg_loss/step
            self.accuracy_test= acc_test/step
            self.accuracy_train = acc_train/step
            
            if self.accuracy_test > self.max_acc:
                self.max_acc = self.accuracy_test
                worse = 0
            else:
                worse += 1
            if cur_epoch % 100 == 0:
                print self.report()

            cur_epoch += 1

            
    def get_pred(self):
        print "getting predictions"
        self.pred_dict = {}
        for batch in self.get_test_batch(self.X_test):
            feed_dict = {self.features : batch}
            pred = sess.run(self.pred, feed_dict)
            
            for i, ind in enumerate(batch.index):
                self.pred_dict[ind] = np.argmax(pred[i])
            
        return self.pred_dict
            
    def get_predictions(self):
        return self.pred
            
            
            
    def report(self):
        return [('loss',self.average_loss), ('test_acc',self.accuracy_test), ('train_acc',self.accuracy_train), ('max_acc', self.max_acc)]



#Ran a hyperparameter search and replaced possible values with tuned parameters
    
epochs = 500
results = []

blist = [32]
lr = [0.0007]
hidden_size = [200]
actiavtion = [tf.nn.tanh]
initializer = [tf.contrib.layers.xavier_initializer()]
scale = [0.3]

hyperparameters = [i for i in itertools.product(blist, lr, hidden_size, actiavtion, initializer, scale)]
shuffle(hyperparameters)

total = len(blist) * len(lr) * len(hidden_size) * len(actiavtion) * len(initializer) * len(scale)
step = 0

#for hyp in hyperparameters:
curavg_loss = 0
curtest_acc = 0
curtrain_acc = 0
max_acc = 0
bs, l, hs, a, i, s = hyperparameters[0]
print ('---------------------------------------------------------')
print ('currently running experiment %d out of %d' % (step, total))
print 'batch size: %d, learning_rate: %f, hidden_size: %d, activation: %s, scale : %f, initializer:' % (bs, l, hs, a.__name__, s)
print i 

# for k in range(3):
 #   print 'running run %d' % (k+1)
param_dict = {'batch_size' : bs, 'lr' : l, 'hidden_size' : hs, 'activation' : a, 'initializer' : i, 'scale' : s}
model = SimpleNeuralNetwork(param_dict, epochs)
data, labels = model.prep_data('data/digits_train.csv', 'data/digits_test.csv')

with tf.Session() as sess:
    sess.as_default()

    model.run(sess)
    report = model.report()
    results.append(report)
    curavg_loss += report[0][1]
    curtest_acc += report[1][1]
    curtrain_acc += report[2][1]
    max_acc += report[3][1]
    print "loss: %f, test_acc: %f, train_acc: %f, max_acc: %f" % (curavg_loss, curtest_acc, curtrain_acc, max_acc)
    preds = model.get_pred()


preds = pd.DataFrame.from_dict(preds, orient='index')
preds.index.name = 'id'
preds.columns = ['pred']

preds.to_csv('final_digits_pred.csv')

