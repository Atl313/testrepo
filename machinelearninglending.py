import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
import scipy.optimize as opt
from sklearn import svm
from sklearn.metrics import f1_score
from sklearn.metrics import jaccard_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss

# This dataset is about past loans. The Loan_train.csv data set includes details of 346 customers whose loan are already paid off or defaulted.
# Loan_status: Whether a loan is paid off on in collection
# Principal: Basic principal loan amount
# Terms: Origination terms which can be weekly (7 days), biweekly, and monthly payoff schedule
# Effective_date: When the loan got originated and took effects
# Due_date: Since it’s one-time payoff schedule, each loan has one single due date
# Age: Age of applicant
# Education: Education of applicant
# Gender: The gender of applicant

# Display settings
pd.options.display.width = 0

df = pd.read_csv('loan_train.csv')
print(df.head())
print(df.shape)

# Convert to date time object
df['due_date'] = pd.to_datetime(df['due_date'])
df['effective_date'] = pd.to_datetime(df['effective_date'])
print(df.head())

# Let’s see how many of each class is in our data set
print(df['loan_status'].value_counts())

# Let's plot some columns to underestand data better:
# By principal
bins = np.linspace(df.Principal.min(), df.Principal.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Paired", col_wrap=2)
g.map(plt.hist, 'Principal', bins=bins, ec="k")
g.axes[-1].legend()
plt.show()

# By age
bins = np.linspace(df.age.min(), df.age.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Paired", col_wrap=2)
g.map(plt.hist, 'age', bins=bins, ec="k")
g.axes[-1].legend()
plt.show()

# Pre-processing: Feature selection/extraction
# Let's look at the day of the week people get the loan
df['dayofweek'] = df['effective_date'].dt.dayofweek
bins = np.linspace(df.dayofweek.min(), df.dayofweek.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Paired", col_wrap=2)
g.map(plt.hist, 'dayofweek', bins=bins, ec="k")
g.axes[-1].legend()
plt.show()

# We see that people who get the loan at the end of the week don't pay it off, so let's use Feature binarization to set a threshold value less than day 4
df['weekend'] = df['dayofweek'].apply(lambda x: 1 if (x > 3) else 0)
print(df.head())

# Convert Categorical features to numerical values
print(df.groupby(['Gender'])['loan_status'].value_counts(normalize=True))

# Let's convert male to 0 and female to 1:
df['Gender'].replace(to_replace=['male', 'female'], value=[0, 1], inplace=True)
print(df.head())

# How about education?
print(df.groupby(['education'])['loan_status'].value_counts(normalize=True))

# Features before One Hot Encoding
print(df[['Principal', 'terms', 'age', 'Gender', 'education']].head())

# Use one hot encoding technique to conver categorical varables to binary variables and append them to the feature Data Frame
Feature = df[['Principal', 'terms', 'age', 'Gender', 'weekend']]
Feature = pd.concat([Feature, pd.get_dummies(df['education'])], axis=1)
Feature.drop(['Master or Above'], axis=1, inplace=True)
print(Feature.head())

# Let's define feature sets, X:
X = Feature
print(X[0:5])

# What are our labels?
y = df['loan_status'].values
print(y[0:5])

# KNN
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
print('Train set:', X_train.shape,  y_train.shape)
print('Test set:', X_test.shape,  y_test.shape)

# Data Standardization give data zero mean and unit variance
X = preprocessing.StandardScaler().fit(X).transform(X)
print(X[0:5])

k = 4
# Train Model and Predict
neigh = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
print(neigh)
yhat = neigh.predict(X_test)
print(yhat[0:5])

# Explore KNN metrics
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))

Ks = 10
mean_acc = np.zeros((Ks - 1))
std_acc = np.zeros((Ks - 1))

for n in range(1, Ks):
    # Train Model and Predict
    neigh = KNeighborsClassifier(n_neighbors=n).fit(X_train, y_train)
    yhat = neigh.predict(X_test)
    mean_acc[n - 1] = metrics.accuracy_score(y_test, yhat)

    std_acc[n - 1] = np.std(yhat == y_test) / np.sqrt(yhat.shape[0])

print(mean_acc)

plt.plot(range(1, Ks), mean_acc, 'g')
plt.fill_between(range(1, Ks), mean_acc - 1 * std_acc, mean_acc + 1 * std_acc, alpha=0.10)
plt.fill_between(range(1, Ks), mean_acc - 3 * std_acc, mean_acc + 3 * std_acc, alpha=0.10, color="green")
plt.legend(('Accuracy ', '+/- 1xstd', '+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.show()

print("The best accuracy was with", mean_acc.max(), "with k =", mean_acc.argmax()+1)

# Decision Tree
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)
X = preprocessing.StandardScaler().fit(X).transform(X)
print(X[0:5])
dTree = DecisionTreeClassifier(criterion="entropy", max_depth=4)
dTree.fit(X_trainset, y_trainset)
predTree = dTree.predict(X_testset)
print(predTree[0:5])
print(y_testset[0:5])
print("Decision Tree Accuracy: ", metrics.accuracy_score(y_testset, predTree))

# Support Vector Machine
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
print('Train set:', X_train.shape, y_train.shape)
print('Test set:', X_test.shape, y_test.shape)
X = preprocessing.StandardScaler().fit(X).transform(X)
print(X[0:5])

clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)
yhat = clf.predict(X_test)
print(yhat[0:5])

# Print SVM metrics
print("SVM Avg F1-score: %.4f" % f1_score(y_test, yhat, average='weighted'))
print("SVM Jaccard score: %.4f" % jaccard_score(y_test, yhat, pos_label='PAIDOFF'))

# Logistic Regression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
print('Train set:', X_train.shape, y_train.shape)
print('Test set:', X_test.shape, y_test.shape)
X = preprocessing.StandardScaler().fit(X).transform(X)
print(X[0:5])

LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train, y_train)
print(LR)

yhat = LR.predict(X_test)
print(yhat)

yhat_prob = LR.predict_proba(X_test)
print(yhat_prob)

# Logistic regression metrics
# jaccard_score(y_test, yhat,pos_label='COLLECTION')
print("LR Jaccard score: : %.2f" % jaccard_score(y_test, yhat, pos_label='PAIDOFF'))
# log_loss(y_test, yhat_prob)
print("LR LogLoss: : %.2f" % log_loss(y_test, yhat_prob))

# Model evaluation using test set

# Load Test set for evaluation
test_df = pd.read_csv('loan_test.csv')
print(test_df.head())

test_df['due_date'] = pd.to_datetime(test_df['due_date'])
test_df['effective_date'] = pd.to_datetime(test_df['effective_date'])
print(test_df.head())

test_df['weekend'] = df['dayofweek'].apply(lambda x: 1 if (x > 3) else 0)
print(test_df.head())

# Preparation
test_df['Gender'].replace(to_replace=['male', 'female'], value=[0, 1], inplace=True)
test_df.head()

Feature = test_df[['Principal', 'terms', 'age', 'Gender', 'weekend']]
Feature = pd.concat([Feature, pd.get_dummies(test_df['education'])], axis=1)
Feature.drop(['Master or Above'], axis=1, inplace=True)
Feature.head()

X = Feature
print(X[0:5])

y = test_df['loan_status'].values
print(y[0:5])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
print('Train set:', X_train.shape, y_train.shape)
print('Test set:', X_test.shape, y_test.shape)

X = preprocessing.StandardScaler().fit(X).transform(X)
print(X[0:5])

# Model and predict KNN
k = 9
neigh = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
print(neigh)
yhat = neigh.predict(X_test)
print(yhat[0:5])

print("KNN Avg F1-score: %.4f" % f1_score(y_test, yhat, average='weighted'))
print("KNN Jaccard score: %.4f" % jaccard_score(y_test, yhat, pos_label='PAIDOFF'))

# Model and predict Decision Tree
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
print('Train set:', X_train.shape,  y_train.shape)
print('Test set:', X_test.shape,  y_test.shape)

X = preprocessing.StandardScaler().fit(X).transform(X)
print(X[0:5])

dTree = DecisionTreeClassifier(criterion="entropy", max_depth=4)
dTree.fit(X_trainset, y_trainset)
predTree = dTree.predict(X_testset)

print(predTree[0:5])
print(y_testset[0:5])

print("Decision Tree Avg F1-score: %.4f" % f1_score(y_test, yhat, average='weighted'))
print("Decision Tree Jaccard score: %.4f" % jaccard_score(y_test, yhat, pos_label='PAIDOFF'))
