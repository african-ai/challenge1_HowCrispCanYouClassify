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




###### Explanation and Methodology
Fairly simple models for both datasets. The Iris dataset is tiny, with around 100 examples for training and only 40 for testing. Large complicated models can have problems with overfitting with such a small dataset. Instead, I used sklearn's RandomForestClassifier because decision trees are not prone to overfitting. With some cross validation, this approach performed well. For the digits dataset, I built a simple one layer neural network. The data seemed to be pretty straightforward so one dense layer seemed to be enough for training, especially after good hyperparameter tuning. 

