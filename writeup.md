#**Traffic Sign Recognition** 

---
You're reading it! and here is a link to my [project code](https://github.com/happyYolanda/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Project Setup
I used the python to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Images and Distribution
Here is an exploratory visualization of the data set. It is a bar chart showing count the number of images for each class

![number_of_images_for_each_class][number_of_images_for_each_class.png]

###Design and Test a Model Architecture

####1. Pre-Pocessing Steps
As a first step, I decided to convert the images to grayscale which is the same as lane line detectin project. 

Here is an example of a traffic sign image after grayscaling:

![only_grayscaled][only_grayscaled.png]

As a second step, I equalized and normalized the image to help the model treating images uniformly

Here is an example of a traffic sign image after normalization:

![only_normalized][only_normalized.png]


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

####3. Model Training
To train the model, I used :

* learning rate of 0.001
* softmax cross entropy as loss function
* AdamOptimizer as optimizer
* batch size of 512, epochs of 50
* 0.6 for dropout in convoution
* 0.5 for dropout in fully connection.

####4. Model Tuning
My final model results were:

* training set accuracy of 0.9836
* validation set accuracy of 0.9444
* test set accuracy of 0.9292

If a well known architecture was chosen:

* The model inspired from Yann Le Cun's [paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf).
* It is a paper about trafic sign recoginization.
* I tried different number of layers and different depth for each convolution layer, and experiment dropout parameters.
 

###Test a Model on New Images

####1. Here are five German traffic signs that I found on the web:

![small_120_kmh_limit][small_120_kmh_limit.png] 

![small_no_vehicles][small_no_vehicles.png] 

![small_priority_road][small_priority_road.png]

![small_road_works][small_road_works.png] 

![small_vehicles_over_3.5_tonnes_prohibited][small_vehicles_over_3.5_tonnes_prohibited.png]

The first image might be difficult to classify because 120 is similar to 20.

####2. Here are the results of the prediction:
Speed limit (120km/h) : Speed limit (20km/h)

Priority road : Speed limit (120km/h)

No vehicles : Priority road

Road work : Road work

Vehicles over 3.5 metric tons prohibited : Vehicles over 3.5 metric tons prohibited


The model was able to correctly guess 2 of the 5 traffic signs, which gives an accuracy of 40%. 

This is lower to the accuracy on the test set of 0.92. The model was overfit for train data.

####3. The top five soft max probabilities were
For the first image : 
.60 : Speed limit (120km/h)

.50 : Speed limit (60km/h)

.40 : Speed limit (20km/h)

.20 : End of all speed and passing limits

.01 : Speed limit (50km/h)


For the second image :
.60 : No vehicles

.50 : Stop

.40 : Speed limit (70km/h)

.20 : Yield

.01 : Road narrows on the right

For the third image :
.60 : Priority road

.50 : No passing

.40 : Speed limit (100km/h)

.20 : No entry

.01 : No vehicles

For the fourth image :
.60 : Road work

.50 : Stop

.40 : Speed limit (80km/h)

.20 : Priority road

.01 : Bicycles crossing

For the fiveth image :
.60 : Roundabout mandatory

.50 : Vehicles over 3.5 metric tons prohibited

.40 : Speed limit (100km/h)

.20 : Priority road

.01 : End of no passing by vehicles over 3.5 metric tons


