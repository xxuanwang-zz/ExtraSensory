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

![Figure 1](https://github.com/xxuan02/ExtraSensory/blob/master/Image/%E5%9B%BE%E7%89%871.png)  
Figure 1: Correlation heatmap of 75 features from Acc sensors. The zoomed image is a detailed map of the first six features. The number in each entry represents the correlation
coefficient. 

We used Sequential Forward Selection and Auto-encoder Feature Selection to select features. 

### 2.1 Sequential Forward Selection  
First, we try a classical wrapping method of feature selection (SFS): sequential forward selection. The wrapping method selects the features by the score of an evaluation algorithm. Here we use the multi-layer perceptron (MLP) model as the evaluation algorithm and use the balanced accuracy to score the performance of the model on the validation set.
* Balanced Accuarcy = (sensitivity+selectivity) / 2  

### 2.2 Auto-encoder Feature Selection  
![Figure 2](https://github.com/xxuan02/ExtraSensory/blob/master/Image/auto.png)  
Figure 2: A diagram of auto-encoder network structure  

SFS is a wrapper based method involving evaluation algorithms, which is quite time-consuming and requires labeled data. As we mentioned before, the 51 context labels involve large portions of unlabeled data and the classes in a label are very unbalanced. These all lead to a poor reliability of selecting the features by the evaluation algorithm. Unsupervised feature selection methods can be more essential in this case. Here we introduce a more efficient unsupervised feature selection method using an auto-encoder network for selecting features with high representability. Figure 2 shows a diagram of such an auto-encoder network.  

![Figure 3](https://github.com/xxuan02/ExtraSensory/blob/master/Image/%E5%9B%BE%E7%89%877.png)  
Figure 3: Comparison between auto-encoder feature selection and SFS and random selection.
![Figure 4](https://github.com/xxuan02/ExtraSensory/blob/master/Image/%E5%9B%BE%E7%89%874.png)  
Figure 4: Distribution of features scores  

We trained the network with all the training data and plotted the distribution of scores of features in Figure 4. The distribution shows that a plenty of features have scores close to zero indicating that they can be represented by other features. The algorithm selects features by scores from high to low ranks. Figure 3 compares the evaluation score of auto-encoder selections using MLP model with SFS and random selections. Auto-encoder selections performs similarly to random selection and overwhelm a little bit when more than 80 features are selected.  

Auto-encoder feature selector is much faster compared to the wrapped methods like SFS with an acceptable performance. So in general, auto-encoder feature selector is definitely an algorithm that deserves to try in this classification problem.  

## 3. COMPARISON OF METHODS
### 3.1 Random Forest Model  
The random forest model is built to set a baseline and therefore compare the result with using neural networks.At the beginning of the modeling process, we only tried to output the six core labels, and it turns out a model with maximum depth of three gives a high accuracy to predict six labels. Then, the prediction extends to all 51 labels, which also includes more instances that have many missing
labels. To deal with such cases, the sample weight is calculated according to the portion of missing labels. At the same time, to deal with the problem of unbalanced classes, we added additional parameters in the model to set it as ‘balanced’. When evaluating the model, we used balanced accuracy as mentioned in the introduction and ignore the missing labels for the calculations.  

![Figure 5](https://github.com/xxuan02/ExtraSensory/blob/master/Image/%E5%9B%BE%E7%89%878.png)  
Figure 5: Random Forest result shows the effectiveness of sample weight (weighted) and balanced class (balanced).  

To see if we really need to use sample weight and balanced class, and also what the depth of the final tree we want, we plot the “Max
Depth vs. Balanced Accuracy” (Figure 5) with all different cases. As the figure shows, either including the balancing parameter to adding the weight will give us much improvement on the performance of the model. As the graph also shows, the random forest model
performs the best when the maximum depth of the result tree is set to be either 21 or 22. When the depth increases furthermore,
the performance on the test set of predicting is even worse due to over-fitting.  

### 3.2 MLP model  
In order to get the best result, we explore three different regularization methods. 1) drop out 2) batch normalization 3) Frobenius
norm on loss function. We add variables to change drop out rate (0 or 0.15), state of batch normalization (On or Off) and regularization
parameter alpha(0 or 0.1 or 0.001) which is related to the Frobenius norm.  
We compared the effect of batch normalization while fixing drop out and alpha as 0, through various numbers of nodes (Figure 6).  
![Figure 6](https://github.com/xxuan02/ExtraSensory/blob/master/Image/%E5%9B%BE%E7%89%879.png)  
Figure 6: Compare batch normalization ’on’ and ’off’ in terms of different hidden layer size.  

In conclusion, among those three, batch normalization showed the biggest improvement by preventing overfitting considering that
we got best Balanced accuracy = 0.895510, with batch normalization =’On’, [256,256]nodes. However, the drop out method and Frobenius norm(regularization parameter) on loss function didn’t show significant improvement compared to batch normalization.
### 3.3 RNN Model  
In the previous paragraph we have introduced the MLP model, but the functions that MLP can achieve are still quite limited. For this Extrasensory data set, the labels have a complex temporal correlation with each other, and the length is various. This relationship is not concerned by MLP. The key point in RNN is that the hidden state of the current network will retain the previous input information
and be used as the output of the current network. We built a simple RNN model to see if there any improvement of performance compared to other models. Similarly by using the instance-weighting matrix, we can ignore the effect of missing labels during training. To implement a basic model in comparison, we added a dense layer, which is the regular deeply connected neural network layer and the most common and frequently used layer, set the activation functionas “sigmoid” together with a 0.2 dropout rate. From this model we got a balanced accuracy around 0.68 from the selected 175 features.  

![Figure 7](https://github.com/xxuan02/ExtraSensory/blob/master/Image/LSTM%20(2).png)  
Figure 7: Comparing simple MLP model to RNN with LSTM model  

In addition to the basic model, we extended it to RNN models with the Long Short Term Memory networks, known as “LSTMs”. Since LSTM has the ability to avoid long-term dependency problems, it might provide a better memory structure for processing the 175 input features. We inserted a LSTM layer into our basic RNN model while keeping all the other hyper-parameters unchanged to see if it can help us to improve our model performance. As a result, the balanced accuracy is higher from the beginning and reaches to about 0.71 with the LSTM layer. We should pay more attention to the training part in the future work in order to get a more accurate and saturate BA.

## 4. CONCLUSIONS  
This project is based on in-the-wild dataset, all the data was collected from users’ regular natural behavior, which means the dataset
has a large scale as well as a rich variability and flexibility. So feature selection occupied a large proportion in our project. After applying sequential forward selection and auto-encoder to the data, we found that balanced accuracy almost saturated when we selected
around 80 features. However, selecting more features doesn’t hurt the performance of MLP models.  

Training deep neural networks is complicated by the fact that the distribution of each layer’s inputs changes during training, as the
parameters of the previous layers change. To solve this problem, we used different parameters and regularizations and found that batch normalization works the best, which also helps to reduce overfitting. We proposed our best MLP model with hyper-parameters of
regularization parameter alpha = 0.0, epoch = 40 and learning rate = 0.0001 with two hidden layers that each has 256 hidden units and two batch normalization layers.
