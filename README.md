# Explore the Deep Learning Models with Extrasensory Dataset
[The ExtraSensory Dataset](http://extrasensory.ucsd.edu/)  

From this data set, our plan is to explore feature selection method among all provided features (225 features), and different methods including MLP(Multi Layer Perceptron), RNN(Recurrent Neural Network) and Classic Machine Learning method(Random Forest) to get better performance in classifying multi-label.
## 1. INTRODUCTION  
This data consists of sensory data and labels and it was collected from 60 participants for approximately 7 days. For each user, it has thousands of instances, typically taken in intervals of 1 minute. Every instance contains measurements from sensors from the user’s personal smartphone, eg. accelerometer, magnetometer, location, and phone-state, as well as an accelerometer and
compass from an additional smartwatch that the data collector provided.  

Based on this model, we tried to modify the construction of the hidden layers, the loss function, hyper-parameters, and additing regularization layers to further improve the model.

## 2. FEATURE SELECTION  
Extrasensory dataset has features extracted from ten types of sensors. Except for phone state sensors which have discrete categories
such as battery state and WiFi availability and the categories have different sources of data, other sensors have many extracted features coming from the same source of time-sequential raw data.  

Taking accelerator as an example, it has 26 features extracted from its raw data and these features may be highly related because of
the homology of the features. Such a high correlation turns out to be very common among all 225 feature pairs, which leads to the necessity of feature learning.  

We used Sequential Forward Selection and Auto-encoder Feature Selection to select features. 

### 2.1 Sequential Forward Selection  
First, we try a classical wrapping method of feature selection (SFS): sequential forward selection. The wrapping method selects the features by the score of an evaluation algorithm. Here we use the multi-layer perceptron (MLP) model as the evaluation algorithm and use the balanced accuracy to score the performance of the model on the validation set.
* Balanced Accuarcy = (sensitivity+selectivity) / 2
### 2.2 Auto-encoder Feature Selection  
SFS is a wrapper based method involving evaluation algorithms, which is quite time-consuming and requires labeled data. As we mentioned before, the 51 context labels involve large portions of unlabeled data and the classes in a label are very unbalanced. These all lead to a poor reliability of selecting the features by the evaluation algorithm. Unsupervised feature selection methods can be more essential in this case. Here we introduce a more efficient unsupervised feature selection method using an auto-encoder network for selecting features with high representability.

Auto-encoder feature selector is much faster compared to the wrapped methods like SFS with an acceptable performance. So in general, auto-encoder feature selector is definitely an algorithm that deserves to try in this classification problem.
## 3. COMPARISON OF METHODS
### 3.1 Random Forest Model
### 3.2 MLP model
### 3.3 RNN Model
## 4. CONCLUSIONS  
This project is based on in-the-wild dataset, all the data was collected from users’ regular natural behavior, which means the dataset
has a large scale as well as a rich variability and flexibility. So feature selection occupied a large proportion in our project. After applying sequential forward selection and auto-encoder to the data, we found that balanced accuracy almost saturated when we selected
around 80 features. However, selecting more features doesn’t hurt the performance of MLP models.

