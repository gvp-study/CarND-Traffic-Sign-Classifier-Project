**Traffic Sign Recognition**


**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/images-histogram.png "Visualization"
[image2]: ./examples/raw-sign.png "Grayscaling"
[image3]: ./examples/normalized-sign.png "Random Noise"
[image4]: ./examples/01.jpg "Traffic Sign 1"
[image5]: ./examples/11.jpg "Traffic Sign 2"
[image6]: ./examples/16.jpg "Traffic Sign 3"
[image7]: ./examples/33.jpg "Traffic Sign 4"
[image8]: ./examples/38.jpg "Traffic Sign 5"
[image9]: ./examples/lenet.png "Architecture"

---

**Writeup / README**

You're reading it! and here is a link to my [project code](https://github.com/gvp-study/CarND-Traffic-Sign-Classifier-Project.git)

**Data Set Summary & Exploration**

I used the pandas library to calculate summary statistics of the traffic
signs data set: The files in pickle format are train.p, valid.p and test.p.

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the training data set. The graph clearly shows the uneven distribution of the histogram of the traffic signs. The first sign (SpeedLimit20) has only about 200 samples while the SpeeLimit50 sign has about 2000 samples. There is a huge variation in the sample set for every sign. Training on such a set will help make the classifier robust.

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I decided to keep all 3 channels of the color image instead of converting it to grayscale. I think the color of the sign could be a useful parameter in the classification and could be learned by the neural network.
To reduce the effects of the lighting on the input vector. I decided to normalize the color images of the traffic signs. I used a simple formula of converting the pixels using the following equation.
pix = (pix - 128)/128.

Here is an example of a traffic sign image before and after normalizing.

![alt text][image2]

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

![alt text][image9]

My final model consists of the following six layers as shown in the figure. Unlike the figure, the inputs in this case are 32x32x3 traffic sign images and the output is a 43 element traffic sign classes.

The inputs is the 32x32x3 image which is fed to the first convolutional layer which convolves the image with a 3x3 kernel with a 1x1 stride. This results in a 32x32x64 output. This is in turn fed through the Rectified Linear Units RELUs to add the nonlinear classification functions. The result of the RELUs is then fed to a softmax pooling layer. The output of the first layer is 14x14x6 is then fed to the second layer which has a similar structure and that layer results in a 5x5x16 output. This output is then flattened into a 400 long vector. The third layer is a fully connected neural net with an output vector 120 long. This is fed through a RELU and softmax functions to output 120 inputs to the fourth layer which is similar in structure and produces 84 outputs. These 84 vector is fed into the last layer which classifies the output into the 43 traffic sign classes.

**Optimizer Type**
I tested the different options for the optimizers. In addition to the original AdamOptimizer, I tried to use GradientDescentOptimizer, RMSPropOptimizer, ProximalGradientDescentOptimizer, MomentumOptimizer. Looking at the rate of convergence of the model, only the RMSPropOptimizer came close to the AdamOptimizer. So, I continued to use the AdamOptimizer. The results are shown below.

No of Epochs = 10
KernelSize = 5x5
Results for the AdamOptimizer(rate=0.001):
Validation Accuracy = 0.560, 0.676, 0.721, 0.760, 0.771, 0.812, 0.813, 0.819, 0.831, 0.828

Results for the RMSPropOptimizer(rate=0.001):
Validation Accuracy = 0.679  0.783, 0.795, 0.832, 0.856, 0.866, 0.871, 0.877, 0.877, 0.882

Results for the ProximalGradientDescentOptimizer(rate=0.001):
Validation Accuracy = 0.057, 0.054, 0.055, 0.056, 0.101, 0.162, 0.260, 0.334, 0.360, 0.436

Results for the MomentumOptimizer(rate=0.001, momentum=0.9):
Validation Accuracy = 0.054, 0.054, 0.054, 0.054, 0.054, 0.054, 0.055, 0.054, 0.054, 0.054

Results for the FtrlOptimizer(rate=0.001):
Validation Accuracy = 0.180, 0.238, 0.304, 0.370, 0.417, 0.448, 0.493, 0.506, 0.543, 0.556

Results for the ProximalAdagradOptimizer(rate=0.001):
Validation Accuracy = 0.165, 0.229, 0.273, 0.306, 0.338, 0.357, 0.386, 0.402, 0.417, 0.439

**BatchSize**
I also tried changing the batch size to see if there is any improvement. I found that increasing the batch size was detrimental to the accuracy. Instead decreasing it to a size of 64 improved the accuracy.
Results for BatchSize = 256 for RMSPropOptimizer(rate=0.001):
Validation Accuracy = 0.359, 0.580, 0.706, 0.785, 0.787, 0.821, 0.820, 0.846, 0.847, 0.860
Results for BatchSize = 256 for AdamOptimizer(rate=0.001):
Validation Accuracy = 0.451, 0.673, 0.734, 0.784, 0.788, 0.806, 0.814, 0.816, 0.823, 0.828

Results for BatchSize = 64 for AdamOptimizer(rate=0.001): Epochs 10
Validation Accuracy = 0.734, 0.821, 0.843, 0.864, 0.868, 0.876, 0.857, 0.824, 0.891, 0.884
Results for BatchSize = 64 for AdamOptimizer(rate=0.001): Epochs 20
Validation Accuracy = 0.813, 0.837, 0.859, 0.851, 0.889, 0.859, 0.869, 0.875, 0.880, 0.883, 0.894, 0.902, 0.901, 0.915, 0.901, 0.883, 0.893

**KernelSize**
I also experimented with the size of the first convolution kernel. I started out with the 5x5 kernel and then tested it with the smaller 3x3 kernel. The 3x3 seems to converge to a better accuracy.
Validation Accuracy = 0.798, 0.843, 0.873, 0.875, 0.890, 0.899, 0.890, 0.886, 0.899, 0.904, 0.906, 0.902, 0.902, 0.89, 0.913, 0.897, 0.888, 0.918, 0.899, 0.914


| Layer         		    |     Description	        			|
|:---------------------:|:---------------------------------------------:|
| Input         		    | 32x32x3 RGB image   					|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					        |								       				  |
| Max pooling	      	  | 2x2 stride,  outputs 14x14x6  |
| Layer2 Input   		    | 14x14x6    					          |
| Convolution 5x5     	| 1x1 stride, same padding, outputs 10x10x16 	|
| RELU					        |								       				  |
| Max pooling	      	  | 2x2 stride,  outputs 5x5x16   |
| Flatten   	      	  |  outputs 400                  |
| Layer3        	      | Input 400, Output 120    									  |
| Fully connected		    | Input 400   									|
| Softmax				        | 120 outputs   								|
| RELU					        |								       				  |
|	Layer4      					|	Input 120, Output 84					|
| Fully connected		    | Input 120   									|
| Softmax				        | 84 outputs   								  |
| RELU					        |								       				  |
|	Layer5      					|	Input 84, Output 43   				|			
| Fully connected		    | Input 84    									|
| Softmax				        | 43 outputs   								  |



####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the AdamOptimizer for the type of optimizer. I also used an Epoch size of 20, a batch size of 64 and a convolution KernelSize of 3x3. I also shuffled the training images and the labels (simultaneously) to prevent any correlation to the order in the inputs.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.870
* validation set accuracy of 0.870
* test set accuracy of 100

I did not use an iterative approach as I was short on time.

I chose the given LeNet solution architecture which recognizes numbers because it was based on an input image and seemed to perform well with the classification the input image into 10 numbers.
The steady rise in the accuracy of the model during a typical training from around 0.673 to a final 0.876 in 10 epochs gives me confidence that the neural network is learning to classify the test data into the 43 traffic signs.
I tried to speed up the learning by increasing the learning rate from 0.001 to 0.01 and the result was not good. The validation accuracy stayed at around 0.05. The value of 0.001 kept the validation accuracy above 0.7 and produced accurate predictions on the test images.

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:


![alt text][image4] ![alt text][image5] ![alt text][image6]
![alt text][image7] ![alt text][image8]

The first thing to notice about these images are their sizes are very different from the standard input image size of 32x32x3 used in for testing and validation of the neural network. So I read in each of these images and used the Python Imaging Library (PIL) to resize the images to the standard input size.

The hardest sign in the sign index 11 (Right-of-way at the next intersection). I suspect this is because there are other signs such as 27 and 20 which have the same triangular shape and color. This sign was the only one which scored less than hundred in the probability for the first guess at 99%. The other four signs seemed to get their first guess probability at 100%.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			            |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 3.5Ton Limit					| 3.5Ton Limit 									    | 100%
| SpeedLimit 30      		| SpeedLimit 30   									| 100%
| KeepRight       			| KeepRight            							| 100%
| Turn-Right-Ahead   		| Turn-Right-Ahead					 				| 100%
| Right-of-way    			| Right-of-way										  | 99%


The model was able to correctly guess all 5 of the traffic signs correctly, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 100%

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 16th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a SpeedLimit30 because of the first guess probability is close to 1.0, and the image does contain a SpeedLimit30. The top three soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.00000000e+00        | 3.5Ton Limit  									  |
| 1.13677868e-19     		| RoundAbout 										    |
| 9.53465566e-21				| NoPassing    										  |

For the second image the corresponding 3 guesses were like so;
| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 9.99855280e-01        | SpeedLimit30  									  |
| 1.44720601e-04     		| SpeedLimit50									    |
| 8.74966044e-10				| WildAnimalsCrossing							  |

For the third image the corresponding 3 guesses were like so;
| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.00000000e+00        | KeepRight  				    					  |
| 1.53114388e-09      	| StraightOrRight   						    |
| 2.21807775e-14				| TurnLeftAhead     							  |

For the third image the corresponding 3 guesses were like so;
| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 9.99362051e-01        | KeepRight  				    					  |
| 5.58210537e-04      	| StraightOrRight   						    |
| 4.78599068e-05				| TurnLeftAhead     							  |

For the fifth image the corresponding 3 guesses were like so;
| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 9.90505278e-01        | Right-of-way  				    			  |
| 9.49474983e-03      	| Pedestrians   			      		    |
| 5.55521229e-09				| DangerousCurveRight							  |

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?