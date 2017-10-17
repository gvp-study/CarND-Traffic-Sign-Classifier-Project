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
[image10]: ./examples/classified-signs.png "Classified-signs"

---

**Code**

This link has my [project code](https://github.com/gvp-study/CarND-Traffic-Sign-Classifier-Project.git)

**Data Set Summary & Exploration**

The files in pickle format are train.p, valid.p and test.p. I used the pandas library to calculate summary statistics of the traffic signs data which is as follows:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43


Here is an exploratory visualization of the data set. It is a bar chart showing how the training data set. The graph clearly shows the uneven distribution of the histogram of the traffic signs. The first sign (SpeedLimit20) has only about 200 samples while the SpeeLimit50 sign has about 2000 samples. There is a huge variation in the sample set for every sign. Training on such a set will help make the classifier robust.

![alt text][image1]

I decided to keep all 3 channels of the color image instead of converting it to grayscale. I think the color of the sign could be a useful parameter in the classification and could be learned by the neural network.
To reduce the effects of the lighting on the input vector. I decided to normalize the color images of the traffic signs. I used a simple formula of converting the pixels using the following normalizing equation.

        pix = (pix - minpix) * 256/maxpix

Here is an example of a traffic sign image before and after normalizing.

![alt text][image2]

![alt text][image9]

My final model consists of the following five main hidden layers as shown in the figure. Unlike the figure, the inputs in this case are 32x32x3 traffic sign images and the output is the 43 element traffic sign classes.

**Layer 1:** Convolution with a 3x3 kernel. The output shape is 28x28x6.

*Activation* RELUs for the nonlinear classification functions

*Pooling* The output of this softmax layer is 14x14x6.

*Layer 2:* Convolution with a 5x5 kernel. The output shape is 10x10x16.
Activation. RELUs for the nonlinear classification functions
Pooling. The output shape is 5x5x16.
Flatten. Using tf.contrib.layers.flatten, we flatten the output shape of the final pooling layer such that it's a 1D vector which has 400 elements.

**Layer 3:** Fully Connected layer. This has 400 outputs.

*Activation* RELUs for the nonlinear classification functions

**Layer 4:** Fully Connected. This has 120 outputs.

*Activation* RELUs for the nonlinear classification functions

**Layer 5:** Fully Connected (Logits). This has 43 outputs.

**Optimizer Types**

I tested the different options for the optimizers. In addition to the original AdamOptimizer, I tried to use GradientDescentOptimizer, RMSPropOptimizer, ProximalGradientDescentOptimizer, MomentumOptimizer. Looking at the rate of convergence of the model, only the RMSPropOptimizer came close to the AdamOptimizer. So, I continued to use the AdamOptimizer. The results are shown below.

Results for the AdamOptimizer(rate=0.001):
Validation Accuracy = 0.828

Results for the RMSPropOptimizer(rate=0.001):
Validation Accuracy = 0.882

Results for the ProximalGradientDescentOptimizer(rate=0.001):
Validation Accuracy = 0.436

Results for the MomentumOptimizer(rate=0.001, momentum=0.9):
Validation Accuracy = 0.054

Results for the FtrlOptimizer(rate=0.001):
Validation Accuracy = 0.556

Results for the ProximalAdagradOptimizer(rate=0.001):
Validation Accuracy = 0.439

**BatchSize**

I also tried changing the batch size to see if there is any improvement. I found that increasing the batch size was detrimental to the accuracy. Instead decreasing it to a size of 64 improved the accuracy.
Results for BatchSize = 256 for RMSPropOptimizer(rate=0.001):
Validation Accuracy =  0.860
Results for BatchSize = 256 for AdamOptimizer(rate=0.001):
Validation Accuracy = 0.828
Results for BatchSize = 64 for AdamOptimizer(rate=0.001): Epochs 10
Validation Accuracy = 0.884
Results for BatchSize = 64 for AdamOptimizer(rate=0.001): Epochs 20
Validation Accuracy =  0.893

**KernelSize**

I also experimented with the size of the first convolution kernel. I started out with the 5x5 kernel and then tested it with the smaller 3x3 kernel. The 3x3 seems to converge to a better accuracy.
Validation Accuracy = 0.914

**Final Parameters**
No of Epochs = 20
BatchSize = 32
KernelSize = 3x3
Lenet:mu = 0.0
Lenet:sigma = 0.01
Learning_rate = 0.001


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
| Flatten   	      	  | Outputs 400                   |
| Layer3        	      | Input 400, Output 120    			|
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


To train the model, I used the AdamOptimizer for the type of optimizer. I also used an Epoch size of 20, a batch size of 64 and a convolution KernelSize of 3x3. I kept the learning_rate at 0.001. I also shuffled the training images and the labels (simultaneously) to prevent any correlation to the order in the inputs.

My final model results were:
* training set accuracy of 0.870
* validation set accuracy of 0.870
* test set accuracy of 100

I did not use an iterative approach as I was short on time.

I chose the given LeNet solution architecture which recognizes numbers because it was based on an input image and seemed to perform well with the classification the input image into 10 numbers.
The steady rise in the accuracy of the model during a typical training from around 0.673 to a final 0.876 in 10 epochs gives me confidence that the neural network is learning to classify the test data into the 43 traffic signs.
I tried to speed up the learning by increasing the learning rate from 0.001 to 0.01 and the result was not good. The validation accuracy stayed at around 0.05. The value of 0.001 kept the validation accuracy above 0.7 and produced accurate predictions on the test images.

**Testing the Model on New Images**

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6]
![alt text][image7] ![alt text][image8]

The first thing to notice about these images are their sizes are very different from the standard input image size of 32x32x3 used in for testing and validation of the neural network. So I read in each of these images and used the Python Imaging Library (PIL) to resize the images to the standard input size.

The hardest sign in the sign index 11 (Right-of-way at the next intersection). I suspect this is because there are other signs such as 27 and 20 which have the same triangular shape and color. This sign was the only one which scored less than hundred in the probability for the first guess at 99%. The other four signs seemed to get their first guess probability at 100%.

Here are the results of the prediction:

| Image			            |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 3.5Ton Limit					| 3.5Ton Limit 									    | 100%
| SpeedLimit 30      		| SpeedLimit 30   									| 100%
| KeepRight       			| KeepRight            							| 100%
| Turn-Right-Ahead   		| Turn-Right-Ahead					 				| 100%
| Right-of-way    			| Right-of-way										  | 99%


The model was able to correctly guess all 5 of the traffic signs correctly, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 100%

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

*Classification of the Five Signs*

I display the final results of the classifcation of the five signs as shown below.

![alt text][image10]

**Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)**

I did not have time to work on this part of the project. I would like to complete it sometime in the coming days.
