#**Traffic Sign Recognition** 

---
You're reading it! and here is a link to my [project code](https://github.com/happyYolanda/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Project Setup
I used the python to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is ?
* The size of test set is ?
* The shape of a traffic sign image is ?
* The number of unique classes/labels in the data set is ?

####2. Images and Distribution
Here is an exploratory visualization of the data set. It is a bar chart showing count the number of images for each class

![alt text][image1]

###Design and Test a Model Architecture

####1. Pre-Pocessing Steps
As a first step, I decided to convert the images to grayscale which is the same as lane line detectin project. 

Here is an example of a traffic sign image before and after grayscaling.

As a second step, I equalized and normalized the image to help the model treating images uniformly

The resulting images look as follows:

![alt text][image2]

####2. Model Architecture
My final model consisted of the following layers:

Input : 32x32x1 grayscale image

Convolution 3x3 : 1x1 stride, valid padding, outputs 32x32x32

RELU

Max pooling : 2x2 stride,  outputs 16x16x32

Convolution 3x3 : 1x1 stride, valid padding, outputs 16x16x64

RELU

Max pooling : 2x2 stride,  outputs 8x8x64

Convolution 3x3 : 1x1 stride, valid padding, outputs 8x8x128

RELU

Max pooling : 2x2 stride,  outputs 4x4x64

Fully connected : 1024 -->  120

Fully connected : 120 --> 119

Fully connected : 119 --> 118
......

Fully connected : 85 --> 84

Fully connected : 84 --> 43

Dropout for convoution : 0.6

Dropout for fully connection : 0.5

####3. Model Training
To train the model, I used learning rate of 0.001, softmax cross entropy as loss function, AdamOptimizer as optimizer, batch size of 512, epochs of 50, 0.6 for dropout in convoution, 0.5 for dropout in fully connection.

####4. Model Tuning
My final model results were:

* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If a well known architecture was chosen:

* The model inspired from Yann Le Cun's [paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf).
* It is a paper about trafic sign recoginization.
* I tried different number of layers and different depth for each convolution layer, and experiment dropout parameters.
 

###Test a Model on New Images

####1. Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Here are the results of the prediction:
Speed limit (120km/h) : 
Priority road : 
No vehicles : 
Road work : 
Vehicles over 3.5 metric tons prohibited : 


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. The top five soft max probabilities were
For the first image : 
.60 : 
.20 : 
.50 :
.40 : 
.01 : 



For the second image :


