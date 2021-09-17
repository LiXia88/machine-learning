#!/usr/bin/env python
# coding: utf-8

# In[9]:


# compare algorithms
#1 Describe what each module and the function used here does.
from pandas import read_csv #"read a comma separated values (csv) file into DataFrame"
from pandas.plotting import scatter_matrix #"able to help us draw a matrix of scatter plots"
from matplotlib import pyplot #"matplotlib import pyplot is a collection of functions that make matplotlib work like MATLAB,"
   # "pyplot allow us make change to a figure.MATLAB is a proprietary multi-paradigm programming language and numerical computing
   # "environment developed by MathWorks."
from sklearn.model_selection import train_test_split #"It's a class which separate arrays or matrices into random train and test subsets"
from sklearn.model_selection import cross_val_score #"Evaluate a score by cross-validation, cross-validation object is"
    #"a variation of KFold that returns stratified folds. The folds are made by preserving the percentage of samples for each class."
from sklearn.model_selection import StratifiedKFold #"Provides train/test indices to split data in train/test sets."
from sklearn.linear_model import LogisticRegression #"This is a class implements regularized logistic regression using the 
    #'liblinear’library. Logistic regression is used to describe data and to explain the relationship between one dependent
    #binary variable and more interval independent variables."
from sklearn.tree import DecisionTreeClassifier #"A tree structure is constructed that breaks the dataset down into smaller subsets 
    #"eventually resulting in a prediction."
from sklearn.neighbors import KNeighborsClassifier #"the KNeighborsClassifier(KNN) looks for the default 5 nearest neighbors"
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis #"A classifier with a linear decision boundary generated ”
    #"by fitting the category conditional density to the data."
from sklearn.naive_bayes import GaussianNB #"Can perform online updates to model parameters via partial_fit."
from sklearn.metrics import classification_report #"Build a text report showing the main classification metrics"
from sklearn.metrics import confusion_matrix #"Compute confusion matrix to evaluate the accuracy of a classification."
from sklearn.metrics import accuracy_score #"this function computes subset accuracy; the set of labels predicted for a sample match 
  #"the corresponding set of labels in y_train."
from sklearn.svm import SVC #"Support Vector Classification is based on libsvm. It's to fit to the data you provide."
#2 What libraries exist within each module. Is this the only module that has such facility?
    #Libraries: Pandas, matplotlib, sklearn. Each modules or functions are only works with exclusive environment/library. 

# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)
#3 Confine the data set to your local machine.
#4 open the file, does it make any sense? Can you use another application to view the content?
    #open the link file you can see the 150 data sets, we can also read it by excel

#shape
print(dataset.shape)
#5 Are there other shape functions that you can draw from the data set?
    #we can also use numpy array shape (np.shape(dataset).

#head
print(dataset.head(150))
#6 What does this mean? Can you read the entire content without passsing a parameter?
   #head() function is used to get the first n row. we have total 150 iris datasets, if we enter head(20), it will only print the first 20
#datasets, and now I enter 150, that means it will print total 150 datasets.If we open the dataset link we can see 150 datasets, but if
#we want to see it in a local machine,then we cannot read it without passing a parameter.

#description
print(dataset.describe())
#7 What does this mean? Where is the description coming from?
#The describe() method is used to calculate some statistical data, such as the percentile, mean and standard deviation of the value of 
    #Series or DataFrame. It's coming from pandas.

#class distribution
print(dataset.groupby('class').size())
#8 What does this mean? Why would pass 'class' as an argument? Can you pass other arguments
#This method is used to compute number of rows in each group as a Series if as_index is True or a DataFrame if as_index is False.
#It would pass 'class' as an argument because we have already claimed 'class' as a value in the previous code "name".
#If we want to pass other arguments, we must claim it in the previous code or change the 'class'to a new argument. 

#box and whisker plots
#dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
pyplot.show()
#9 Change the attributes for the box and whisker plots - I changed the box to line

#histograms
dataset.hist()
pyplot.show()
# scatter plot matrix
scatter_matrix(dataset)
pyplot.show()
#10 What other plotting methods can you show?
    #we can use mateplotlib.plot to draw lines on x-axis and y-axis graph

# Split-out validation dataset
array = dataset.values
X = array[:,0:4]
y = array[:,4]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1, shuffle=True)
#11 What does this mean? Can you change the array and show different validation
    #in the previous, we imported a class "train_test_split",using this we can easily split the dataset into the training 
#and the testing datasets in various proportions. X_train and Y_train is the trains where we are split to. 
    #test_size=0.20 means we are training the model with 80% data, leaving 20%(30 obersvations) for validation.
    #random_state is help verifying the output. It's make sure that you obtain the same split everytime you run the code, the 
#number of interger is doesn't matter everytime the result will be the same value in the train and test dataset.
#the test_size and random_state can be changed, but the value of x and y array cannot be changed. 4 is the a constant value

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
#12 What other learning algorithm modules are contained within the library here?
    #we also used liblinear and ovr.
    #OvR(one vs restclassifier) is a strategy involves equipping each class with a classifier. For each classifier, 
#the classification will fit all other classifications. Since each class is represented by one classifier,
#knowledge about the class can be obtained by checking its corresponding classifier. This is the most common strategy 
#used for multi-class classification and a reasonable default choice.
    #Liblinear solver supports both L1 and L2 regularization, with a dual formulation only for the L2 penalty. 
#LIBLINEAR is a linear classifier for data with millions of instances and features. It supports: L2-regularized classifiers; 
#L2-loss linear SVM, L1-loss linear SVM, and logistic regression L1-regularized classifiers.

# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
#13 what algorithms we used here? why we use this function here?
# We used stratifiedKFold to split data into train and test datasets.
# We also used for-loop to print the results of spot check algorithms
# %s is used as a placeholder for string values you want to inject into a formatted string.
#  f-string is a literal string, prefixed with 'f', which contains expressions inside brace. 
# there is other operator %d which used a placeholder for decimal or numeric values

# Compare Algorithms
pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm Comparison')
pyplot.show()
#14 what is this mean? what library we used here?
# We used matplotlib to show the result of algorithm in the figure.

# Make predictions on validation dataset
model=SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)
#Evaluate predictions
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
#15 what is this mean?
#accuracy_score is the percent of the validation set that was correctly predicted. The validation set 
#is 30 observations and there was 1 error so the accuracy score is 0.9666 because 96% of the observations 
#in the validation set were correctly classified.
#confusion_matrix is showing the observations that were misclassified.
#classification report has 3 parts for each grouping in the data:
#precision: the ability of the classifier is to label as negative a sample 
#recall: the ability of the classifier to find all positive samples
#f1-score: the harmonic mean of precision and recall.
#It also shows support, support is the number of observations in the validation set belonging to each class


# In[ ]:




