# How Crisp Can you Classify? : A Data Challenge to flex your classification skills 

## Data Challenge Task
This Data Challenge has 2 tasks;
* to determine the type of an Iris plant given data on sepal and petal lengths and widths of the flowers

* to identify a digit/number given optical data from handwritten digits

## Data Challenge Bounty
### USD 100 
to the highest ranked model submission, based on the Git timestamps 
### USD 10 
to the next top 10 ranked submissions, based on Git timestamps
### Invitation to an Exclusive Data Hackathon 
Where we will hack on a real business’ data and stand a chance to win even more exciting prizes and learn from experienced mentors 
[this is for those based in or willing to travel to Nairobi]

## Due Date: August 10th

###### Rules of Engagement
The Data Challenge is judged based on the following criteria:
  - A Correct fork, branch and pull request
  - Using the GitHub Pull Request timestamp where order of submissions is applicable
  - Using solution quality/accuracy and explanation to rank submissions where applicable 
  - Do not share any code that you cannot open source on the Git Repository as its open source and african.ai will not be liable for any breach of intellectual property (if at all) once shared on the platform.

## Working on the Data Challenge
1.Fork the code challenge repository provided.

2.Make a topic branch. In your github form, keep the master branch clean. When you create a branch, it essentially will be a copy of the master.

>Pull all changes, make sure your repository is up to date

```sh
$ cd challenge1_HowCrispCanYouClassify
$ git pull origin master
```

>Create a new branch as follows-> git checkout -b [your_phone_number_email], e.g.

```sh
$ git checkout -b 2348177779360_youremail@gmail.com master
```

>See all branches created

```sh
$ git branch
* 2348177779360_youremail@gmail.com
  master
```

>Push the new branch to github

```sh
$ git push origin -u 2348177779360_youremail@gmail.com
```

3.**Remember to only make changes to the fork!**

The folder named **data** contains 4 csv files. 
* iris_train
* iris_test
* digits_train
* digits_test

The folder names **submissions** contains 2 csv files.
* digits_sample_submission
* iris_sample_submission

The train datasets contain labelled records, ie. their classes are known.
In each case:
* use the train datasets to train a satisfactory classification model
* use the model to classify the records in the test datasets
* ensure the format of your submission files is similar to the 
* once satisfied with the model and the predictions, name the file containing labelled test data **iris_test_labelled** or **digit_test_labelled** and include it in the **submissions** folder, submission files should include only 2 columns, the id and the predicted labels
* Add to the base of the existing README file a brief explanation about your solution outlining the algorithm you chose to use, why you chose it and how the algorithm compared to any others you may have tried to use  

4.Commit the changes to your fork.

5.Make a pull request to the **challenge1_HowCrispCanYouClassify** Repo.


##### Dataset Details

The Iris Dataset Details:
* 150 instances/records
	* train - 110 records
	* test  - 40 records
* 4 attributes/features
* 3 classes

The Handwritten Digits Dataset details: 
* 5620 instances/records 
	* train - 4000 records
	* test  - 1620 records
* 64 attributes/features
* 10 classes

 
##### Resources
You can use the following resources to to get acquainted with some classification problems:
* [Naive Bayes Classification](https://github.com/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/05.05-Naive-Bayes.ipynb)
* [Support Vector Machines](https://github.com/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/05.07-Support-Vector-Machines.ipynb)

###### Terms and Conditions
  - Each Data Hacker can participate in as many challenges as they wish
  - Branches per Data Challenge need to be a unique combination of a genuine phone number and email address ```2348177779360_youremail@gmail.com```
  - Multiple submissions are allowed for as long as the challenge is still open, once the challenge is closed, the last submitted changes will be the evaluated solution
  - african.ai reserves the right to announce the winners
  - african.ai reserves the right to reward the winners based on african.ai criterion
  - Do not share any code that you cannot open source on the Git Repository as it is public and african.ai will not be liable for any breach of intellectual property (if any) once shared on the platform.
  - Data Challenges are time bound - the time restriction is specified on each challenge
  - Additional rules MAY be provided on the code challenge and will vary for each challenge
  - You are free to use all manner of tools
  - Successive interviews for projects MAY be run to satisfy participating african.ai partners


## FLOWER CLASSIFIER: A CELL-BY-CELL WALKTHROUGH
Decided to go with tensorflow's DNN Classifier for this problem. 

A simple linear model classifier would have done for this one, but I tried using it and felt I should challenge myself with something else.
Here's the workflow on how I handled the flower classification problem:

```python
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

```


```python
train_df = pd.read_csv('data/iris_train.csv', skiprows=[0], usecols=[1,2,3,4,5], header=None)
test_df = pd.read_csv('data/iris_test.csv', skiprows=[0], usecols=[1,2,3,4], header=None)
test_array = np.array(test_df, dtype=np.float32)
actual_data = pd.read_csv("data/iris_train.csv")
# reset the column index after omitting the first unnecessary columns
train_df = train_df.T.reset_index(drop=True).T
test_df = test_df.T.reset_index(drop=True).T

labels_data = pd.read_csv('data/iris_train.csv', usecols=[5], skiprows=[0], header=None)
labels = np.array(np.unique(labels_data), 'str')

```


```python
# iterate through each row, changing the last column (4) to have integers instead of flower names
for index, row in train_df.iterrows():
    if row[4] == labels[0]:
        train_df.loc[index, 4] = 0
    elif row[4] == labels[1]:
        train_df.loc[index, 4] = 1
    else:
        train_df.loc[index, 4] = 2
test_df.head()

```


<div>
<style>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7.2</td>
      <td>3.6</td>
      <td>6.1</td>
      <td>2.5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.4</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.5</td>
      <td>2.3</td>
      <td>1.3</td>
      <td>0.3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6.4</td>
      <td>3.2</td>
      <td>4.5</td>
      <td>1.5</td>
    </tr>
  </tbody>
</table>
</div>




```python
# shuffle the newly formatted data
data = train_df.sample(frac=1).reset_index(drop=True)
y = actual_data.labels

# split that data, 80-20 rule
X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.05)
print (X_test.head())

# prepare data to be consumed by tensorflow
training_header = list((len(X_train), len(X_train.columns) - 1)) + list(labels)    
testing_header = list((len(X_test), len(X_test.columns) - 1)) + list(labels)
print (training_header)

# write the formatted data to csv
X_train.to_csv('training_data_formatted.csv', header=training_header, index=False, index_label=False)
X_test.to_csv('testing_data_formatted.csv', header=testing_header, index=False, index_label=False)
#test_df.to_csv('testing_data_formatted.csv', header=testing_header, index=False, index_label=False)

```

           0    1    2    3  4
    14   5.9    3  5.1  1.8  2
    95   5.6  2.7  4.2  1.3  1
    56   5.5  2.4  3.7    1  1
    78   6.7    3  5.2  2.3  2
    107  4.8  3.1  1.6  0.2  0
    [104, 4, 'Iris-setosa', 'Iris-versicolor', 'Iris-virginica']



```python
# enter tensorflow
from tensorflow.contrib.learn.python.learn.datasets import base

training_set = base.load_csv_with_header(filename='training_data_formatted.csv', features_dtype=np.float32, target_dtype=np.int)
testing_set = base.load_csv_with_header(filename='testing_data_formatted.csv', features_dtype=np.float32, target_dtype=np.int)


```


```python
# Specify that all feature columns have real-value data
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]
```


```python
classifier = tf.contrib.learn.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=[10, 20, 10],
    n_classes=3,
    model_dir="/tmp/iris_model"
)
```

    INFO:tensorflow:Using default config.
    INFO:tensorflow:Using config: {'_task_type': None, '_task_id': 0, '_cluster_spec': <tensorflow.python.training.server_lib.ClusterSpec object at 0x1a20038860>, '_master': '', '_num_ps_replicas': 0, '_num_worker_replicas': 0, '_environment': 'local', '_is_chief': True, '_evaluation_master': '', '_train_distribute': None, '_device_fn': None, '_tf_config': gpu_options {
      per_process_gpu_memory_fraction: 1.0
    }
    , '_tf_random_seed': None, '_save_summary_steps': 100, '_save_checkpoints_secs': 600, '_log_step_count_steps': 100, '_session_config': None, '_save_checkpoints_steps': None, '_keep_checkpoint_max': 5, '_keep_checkpoint_every_n_hours': 10000, '_model_dir': '/tmp/iris_model'}



```python
# define the training inputs
def training_inputs():
    x = tf.constant(training_set.data)
    y = tf.constant(training_set.target)
    
    return x,y
```


```python
# fit the classifier with the training data
classifier.fit(input_fn=training_inputs, steps=2000)
```

    INFO:tensorflow:Create CheckpointSaverHook.
    INFO:tensorflow:Graph was finalized.
    INFO:tensorflow:Restoring parameters from /tmp/iris_model/model.ckpt-20000
    INFO:tensorflow:Running local_init_op.
    INFO:tensorflow:Done running local_init_op.
    INFO:tensorflow:Saving checkpoints for 20000 into /tmp/iris_model/model.ckpt.
    INFO:tensorflow:loss = 0.008319289, step = 20001
    INFO:tensorflow:global_step/sec: 912.958
    INFO:tensorflow:loss = 0.0074323323, step = 20101 (0.111 sec)
    INFO:tensorflow:global_step/sec: 1222.45
    INFO:tensorflow:loss = 0.007272432, step = 20201 (0.084 sec)
    INFO:tensorflow:global_step/sec: 1133.48
    INFO:tensorflow:loss = 0.0071174204, step = 20301 (0.086 sec)
    INFO:tensorflow:global_step/sec: 1279.89
    INFO:tensorflow:loss = 0.006966722, step = 20401 (0.079 sec)
    INFO:tensorflow:global_step/sec: 1249.12
    INFO:tensorflow:loss = 0.006820103, step = 20501 (0.080 sec)
    INFO:tensorflow:global_step/sec: 1357.5
    INFO:tensorflow:loss = 0.006677349, step = 20601 (0.073 sec)
    INFO:tensorflow:global_step/sec: 1233.37
    INFO:tensorflow:loss = 0.006538292, step = 20701 (0.082 sec)
    INFO:tensorflow:global_step/sec: 941.831
    INFO:tensorflow:loss = 0.00640277, step = 20801 (0.106 sec)
    INFO:tensorflow:global_step/sec: 1223.1
    INFO:tensorflow:loss = 0.0062722648, step = 20901 (0.082 sec)
    INFO:tensorflow:global_step/sec: 1253.56
    INFO:tensorflow:loss = 0.05075315, step = 21001 (0.080 sec)
    INFO:tensorflow:global_step/sec: 1301.35
    INFO:tensorflow:loss = 0.006317268, step = 21101 (0.078 sec)
    INFO:tensorflow:global_step/sec: 1153.78
    INFO:tensorflow:loss = 0.006188334, step = 21201 (0.085 sec)
    INFO:tensorflow:global_step/sec: 1090.45
    INFO:tensorflow:loss = 0.006065223, step = 21301 (0.092 sec)
    INFO:tensorflow:global_step/sec: 1169.15
    INFO:tensorflow:loss = 0.0059461007, step = 21401 (0.086 sec)
    INFO:tensorflow:global_step/sec: 1039.09
    INFO:tensorflow:loss = 0.0058302516, step = 21501 (0.098 sec)
    INFO:tensorflow:global_step/sec: 1077.86
    INFO:tensorflow:loss = 0.005717322, step = 21601 (0.092 sec)
    INFO:tensorflow:global_step/sec: 1125.61
    INFO:tensorflow:loss = 0.0056071845, step = 21701 (0.089 sec)
    INFO:tensorflow:global_step/sec: 1133.4
    INFO:tensorflow:loss = 0.005499768, step = 21801 (0.088 sec)
    INFO:tensorflow:global_step/sec: 1042.69
    INFO:tensorflow:loss = 0.0053948984, step = 21901 (0.095 sec)
    INFO:tensorflow:Saving checkpoints for 22000 into /tmp/iris_model/model.ckpt.
    INFO:tensorflow:Loss for final step: 0.0052935258.





    DNNClassifier(params={'head': <tensorflow.contrib.learn.python.learn.estimators.head._MultiClassHead object at 0x1c2b147fd0>, 'hidden_units': [10, 20, 10], 'feature_columns': (_RealValuedColumn(column_name='', dimension=4, default_value=None, dtype=tf.float32, normalizer=None),), 'optimizer': None, 'activation_fn': <function relu at 0x11b8fd9d8>, 'dropout': None, 'gradient_clip_norm': None, 'embedding_lr_multipliers': None, 'input_layer_min_slice_size': None})




```python
# Define test inputs
def test_inputs():
    x = tf.constant(testing_set.data)
    y = tf.constant(testing_set.target)
    
    return x, y
```


```python
# evaluate the classifier's accuracy
accuracy_score = classifier.evaluate(input_fn=test_inputs, steps=1)['accuracy']
print ("Accuracy score", accuracy_score)
```

    INFO:tensorflow:Starting evaluation at 2018-07-29-09:27:17
    INFO:tensorflow:Graph was finalized.
    INFO:tensorflow:Restoring parameters from /tmp/iris_model/model.ckpt-22000
    INFO:tensorflow:Running local_init_op.
    INFO:tensorflow:Done running local_init_op.
    INFO:tensorflow:Evaluation [1/1]
    INFO:tensorflow:Finished evaluation at 2018-07-29-09:27:17
    INFO:tensorflow:Saving dict for global step 22000: accuracy = 1.0, global_step = 22000, loss = 0.0015137732
    Accuracy score 1.0



```python
# classify the new flower samples
def new_flower_samples():
    # take in the test file
    return test_array

# predict the type of flower
predictions = classifier.predict_classes(input_fn=new_flower_samples)
results = list(predictions)
print(results)
```

    INFO:tensorflow:Graph was finalized.
    INFO:tensorflow:Restoring parameters from /tmp/iris_model/model.ckpt-22000
    INFO:tensorflow:Running local_init_op.
    INFO:tensorflow:Done running local_init_op.
    [2, 0, 0, 0, 1, 1, 0, 1, 1, 2, 1, 2, 2, 1, 1, 0, 2, 2, 2, 2, 0, 0, 0, 1, 0, 2, 0, 1, 1, 2, 1, 2, 0, 0, 1, 1, 1, 2, 1, 0]



```python
print(results)
```

    [2, 0, 0, 0, 1, 1, 0, 1, 1, 2, 1, 2, 2, 1, 1, 0, 2, 2, 2, 2, 0, 0, 0, 1, 0, 2, 0, 1, 1, 2, 1, 2, 0, 0, 1, 1, 1, 2, 1, 0]



```python
## create a DataFrame and export it to a csv after formatting the outputs
df = pd.DataFrame(results)

# iterate through each row, changing the last column (2) to have flower-names instead of integers
for index, row in df.iterrows():
    if row[0] == 0:
        df.loc[index, 0] = labels[0]
    elif row[0] == 1:
        df.loc[index, 0] = labels[1]
    else:
        df.loc[index, 0] = labels[2]

df.to_csv('submissions/iris_test_labelled.csv')
```

## DIGIT CLASSIFIER SOLUTION
I created a NN with two fully-contencted layers using Keras. At the 2nd layer, I utilized a softmax function that helped me to get a score for each digit(0-9) –– so each score is the probability that the current digit image belongs to one of the 10 digit classes. The probabilities all summed to 1. 

Here's how my notebook looked like:

```python
import pandas as pd
import numpy as np
from keras import models, layers
```

    /Users/jee/.pyenv/versions/anaconda3-5.2.0/lib/python3.6/importlib/_bootstrap.py:219: RuntimeWarning: numpy.dtype size changed, may indicate binary incompatibility. Expected 96, got 88
      return f(*args, **kwds)
    /Users/jee/.pyenv/versions/anaconda3-5.2.0/lib/python3.6/importlib/_bootstrap.py:219: RuntimeWarning: numpy.dtype size changed, may indicate binary incompatibility. Expected 96, got 88
      return f(*args, **kwds)
    /Users/jee/.pyenv/versions/anaconda3-5.2.0/lib/python3.6/importlib/_bootstrap.py:219: RuntimeWarning: numpy.dtype size changed, may indicate binary incompatibility. Expected 96, got 88
      return f(*args, **kwds)
    /Users/jee/.pyenv/versions/anaconda3-5.2.0/lib/python3.6/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
      from ._conv import register_converters as _register_converters
    Using TensorFlow backend.
    /Users/jee/.pyenv/versions/anaconda3-5.2.0/lib/python3.6/importlib/_bootstrap.py:219: RuntimeWarning: numpy.dtype size changed, may indicate binary incompatibility. Expected 96, got 88
      return f(*args, **kwds)
    /Users/jee/.pyenv/versions/anaconda3-5.2.0/lib/python3.6/importlib/_bootstrap.py:219: RuntimeWarning: numpy.dtype size changed, may indicate binary incompatibility. Expected 96, got 88
      return f(*args, **kwds)



```python
#prepare the data --- one hot encode it for the model to consume
from keras.utils import to_categorical

training_columns = range(1, 65)
testing_columns = range(1, 65)
raw_data = pd.read_csv('data/digits_train.csv', skiprows=[0], usecols=training_columns, header=None)
training_data = raw_data.T.reset_index(drop=True).T

testing_data = training_data[3900:]
testing_data = testing_data.T.reset_index(drop=True).T

training_data = training_data[:3900]
training_data = training_data.T.reset_index(drop=True).T

y = pd.read_csv('data/digits_train.csv', skiprows=[0], usecols=[65], header=None)
y = y.T.reset_index(drop=True).T
y_test = y[3900:]
y_train = y[:3900]

# one hot encode training data and labels
training_images = to_categorical(training_data)
training_labels = to_categorical(y_train)
testing_images = to_categorical(testing_data)
test_labels = to_categorical(y_test)

print(training_images.shape)
print(testing_images.shape)
# reshape the training images in a shape the network will understand
training_images = training_images.reshape(3900, 64 * 17)
training_images = training_images.astype('float32') / 255
testing_images = testing_images.reshape(100, 64 * 17)
testing_images = testing_images.astype('float32') / 255
print(testing_images.shape)
```

    (3900, 64, 17)
    (100, 64, 17)
    (100, 1088)



```python

# create a network with two fully-contencted(dense) layers.
# the 2nd layer return an array of 10 probability scores summing to 1.
# each score is the probability that the current digit image belongs to one of the 10 digit classes.

network = models.Sequential()
network.add(layers.Dense(512, activation="relu", input_shape=(64 * 17,)))
network.add(layers.Dense(10, activation="softmax"))

# compile the network by specifying the loss function, the optimizer and metrics to monitor during training
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
```


```python
# fit the model to the data
network.fit(training_images, training_labels, epochs=30, batch_size=128)
```

    Epoch 1/30
    3900/3900 [==============================] - 1s 133us/step - loss: 0.1232 - acc: 0.9777
    Epoch 2/30
    3900/3900 [==============================] - 0s 81us/step - loss: 0.1195 - acc: 0.9790
    Epoch 3/30
    3900/3900 [==============================] - 0s 90us/step - loss: 0.1144 - acc: 0.9810
    Epoch 4/30
    3900/3900 [==============================] - 0s 86us/step - loss: 0.1109 - acc: 0.9803
    Epoch 5/30
    3900/3900 [==============================] - 0s 73us/step - loss: 0.1069 - acc: 0.9823
    Epoch 6/30
    3900/3900 [==============================] - 0s 64us/step - loss: 0.1034 - acc: 0.9831
    Epoch 7/30
    3900/3900 [==============================] - 0s 80us/step - loss: 0.1001 - acc: 0.9838
    Epoch 8/30
    3900/3900 [==============================] - 0s 75us/step - loss: 0.0965 - acc: 0.9846
    Epoch 9/30
    3900/3900 [==============================] - 0s 81us/step - loss: 0.0932 - acc: 0.9862
    Epoch 10/30
    3900/3900 [==============================] - 0s 68us/step - loss: 0.0897 - acc: 0.9859
    Epoch 11/30
    3900/3900 [==============================] - 0s 79us/step - loss: 0.0870 - acc: 0.9872
    Epoch 12/30
    3900/3900 [==============================] - 0s 77us/step - loss: 0.0841 - acc: 0.9877
    Epoch 13/30
    3900/3900 [==============================] - 0s 84us/step - loss: 0.0812 - acc: 0.9885
    Epoch 14/30
    3900/3900 [==============================] - 0s 71us/step - loss: 0.0784 - acc: 0.9882
    Epoch 15/30
    3900/3900 [==============================] - 1s 208us/step - loss: 0.0757 - acc: 0.9882
    Epoch 16/30
    3900/3900 [==============================] - 1s 200us/step - loss: 0.0735 - acc: 0.9887
    Epoch 17/30
    3900/3900 [==============================] - 1s 204us/step - loss: 0.0708 - acc: 0.9915
    Epoch 18/30
    3900/3900 [==============================] - 1s 179us/step - loss: 0.0689 - acc: 0.9903
    Epoch 19/30
    3900/3900 [==============================] - 1s 176us/step - loss: 0.0665 - acc: 0.9905
    Epoch 20/30
    3900/3900 [==============================] - 1s 178us/step - loss: 0.0641 - acc: 0.9913
    Epoch 21/30
    3900/3900 [==============================] - 1s 189us/step - loss: 0.0618 - acc: 0.9915
    Epoch 22/30
    3900/3900 [==============================] - 1s 173us/step - loss: 0.0595 - acc: 0.9928
    Epoch 23/30
    3900/3900 [==============================] - 1s 178us/step - loss: 0.0572 - acc: 0.9931
    Epoch 24/30
    3900/3900 [==============================] - 1s 165us/step - loss: 0.0556 - acc: 0.9931
    Epoch 25/30
    3900/3900 [==============================] - 0s 81us/step - loss: 0.0539 - acc: 0.9931
    Epoch 26/30
    3900/3900 [==============================] - 0s 77us/step - loss: 0.0521 - acc: 0.9931
    Epoch 27/30
    3900/3900 [==============================] - 0s 74us/step - loss: 0.0503 - acc: 0.9944
    Epoch 28/30
    3900/3900 [==============================] - 1s 163us/step - loss: 0.0484 - acc: 0.9941
    Epoch 29/30
    3900/3900 [==============================] - 1s 210us/step - loss: 0.0469 - acc: 0.9946
    Epoch 30/30
    3900/3900 [==============================] - 1s 191us/step - loss: 0.0453 - acc: 0.9954





    <keras.callbacks.History at 0xb2affe898>




```python
# evaluate the accuracy of the model
test_loss, test_accuracy = network.evaluate(testing_images, test_labels)
print(test_accuracy)
```

    100/100 [==============================] - 0s 114us/step
    0.96



```python
# time to find predictions
data = pd.read_csv('data/digits_test.csv', skiprows=[0], usecols=testing_columns, header=None)
data.head()
# format it to be ingested for prediction
test_data = to_categorical(data)
test_data.shape
test_data = test_data.reshape(1620, 64 * 17)
test_data.astype('float32') / 255

# make predictions
pred = network.predict_classes(test_data)
results = list(pred)
print(results[:10])
```

    [6, 3, 9, 2, 5, 7, 6, 3, 1, 0]



```python
# create a dataframe containing the results
df = pd.DataFrame(results)
```

```python
# write the predictions to output file
df.to_csv('submissions/digits_test_labelled.csv', index_label='id', header=['labels'])
```
