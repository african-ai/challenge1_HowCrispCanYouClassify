from sklearn.ensemble import RandomForestClassifier

train_data_x = []
train_data_y = []

test_data_x = []

train_file = open("iris_train.csv", "r")
test_file = open("iris_test.csv", "r")

train_lines = train_file.readlines()
test_lines = test_file.readlines()

del train_lines[0]
del test_lines[0]

for line in train_lines:
    line = line.split(",")
    train_data_x.append([float(line[1]), float(line[2]), float(line[3]), float(line[4])])
    train_data_y.append(line[5].strip())

for line in test_lines:
    line = line.split(",")
    test_data_x.append([float(line[1]), float(line[2]), float(line[3]), float(line[4])])

clf = RandomForestClassifier(max_depth=3, random_state=42)
clf.fit(train_data_x, train_data_y)

pred = clf.predict(test_data_x)


with open("../submissions/iris_submission.csv", "w") as submission:
    for i in range(pred.shape[0]):
        submission.write(str(i)+","+pred[i]+"\n")




    
