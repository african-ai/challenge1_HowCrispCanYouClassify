import numpy as np

from keras.utils import np_utils
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.layers.core import Dense, Dropout, Activation

train_file = open("digits_train.csv", "r")
test_file = open("digits_test.csv", "r")

train_lines = train_file.readlines()
test_lines = test_file.readlines()

del train_lines[0]
del test_lines[0]

train_x = []
train_y = []

test_x = []

for line in train_lines:
    line = line.split(",")
    part = []
    for i in range(1, len(line)-1):
	    part.append(int(line[i]))
    train_x.append(part)
    train_y.append(int(line[len(line)-1].strip()))

for line in test_lines:
    line = line.split(",")
    part = []
    for i in range(1, len(line)):
	    part.append(int(line[i]))
    test_x.append(part)

train_y = np_utils.to_categorical(train_y, 10)

model = Sequential()
model.add(Dense(512, input_shape=(64,)))
model.add(Activation('relu')) 
model.add(Dropout(0.2))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam',  metrics=['accuracy'])

model.fit(np.asarray(train_x), np.asarray(train_y),
          batch_size=128, nb_epoch=7)

pred = model.predict(np.asarray(test_x)).tolist()

with open("../submissions/digits_submission.csv", "w") as file:
    for i in range(len(pred)):
        file.write(str(i)+","+str(pred[i].index(max(pred[i])))+"\n")

        




