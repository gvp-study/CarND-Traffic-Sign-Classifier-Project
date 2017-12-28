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
[image11]: ./examples/CNN-layer1.png "CNN-layer1"
[image12]: ./examples/CNN-layer2.png "CNN-layer2"
[image13]: ./examples/CNN-layer3.png "CNN-layer3"
[image14]: ./examples/Hidden-Layers.png "CNN-image"
[image15]: ./examples/Accuracy-plot.png "Accuracy-plot"
[image16]: ./examples/Histogram-filling.png "Histogram-filling"
[image17]: ./examples/Classified-graphically.png "Classified-graphically"
[image18]: ./examples/Translate-rotate-bright.png "Translate-rotate-bright"

---
**Modified Resubmit**
This document incorporates all the recommendations of the first reviewer. I had to take an emergency leave of absence due to a death in the family. I got a deferment of the course for 2 months till December 15 2017.

**Code**

This link has my modified [project code](https://github.com/gvp-study/CarND-Traffic-Sign-Classifier-Project6.git)

**Data Set Summary & Exploration**

The files in pickle format are train.p, valid.p and test.p. I used the pandas library to calculate summary statistics of the traffic signs data which is as follows:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43


Here is an exploratory visualization of the data set. It is a bar chart showing how the training data set. The graph clearly shows the uneven distribution of the histogram of the traffic signs. The first sign (SpeedLimit20) has only about 200 samples while the SpeeLimit50 sign has about 2000 samples. There is a huge variation in the sample set for every sign. Training on such a set will help make the classifier robust.

![alt text][image1]

In the orginal submission I decided to keep all 3 channels of the color image instead of converting it to grayscale. But, according to suggestion from the reviewer and the results of experiments, I decided to convert the color image to a grayscale image. I also decided to normalize the color images of the traffic signs using a simple formula of converting the pixels using the following equation.

        pix = (pix - 128) / 128

In my original submission, I did not add any new training samples and the histogram plot of the 43 traffic signs stayed uneven, potentially biasing the results of the classifier. To reduce this, I added the functions to randomly translate, rotate and brighten the a given image. I then took all the traffic signs which had low sample counts in the histogram. I then randomly sampled these sign training example images and randomly translated, rotated and brightened them. These modified images were then added back into the training samples for that traffic sign. The effect of the translate, rotate and brighten on a sample image is shown below.

![alt text][image18]

The result of the adding the additional randomly modified images to the histogram is shown below. Note that the histogram now clearly shows that no traffic sign has a count less than 600. I think this helped improve the accuracy of my system.

![alt text][image16]

My final model consists of the following five main hidden layers as shown in the figure. Unlike the figure, the inputs in this case are 32x32x1 traffic sign images and the output is the 43 element traffic sign classes.
The main change is the input is now a grayscale image unlike the color image in the previous submission. I also changed the first convolution back to a 5x5 kernel.

![alt text][image9]

**Layer 1:** Convolution with a 5x5 kernel. The output shape is 28x28x6.

*Activation* RELUs for the nonlinear classification functions

*Pooling* The output of this softmax layer is 14x14x6.

*Layer 2:* Convolution with a 5x5 kernel. The output shape is 10x10x16.
Activation. RELUs for the nonlinear classification functions
Pooling. The output shape is 5x5x16.
Flatten. Using the tensorflow function tf.contrib.layers.flatten, we flatten the output shape of the final pooling layer such that it's a 1D vector which has 400 elements.

**Layer 3:** Fully Connected layer. This has 400 outputs.

*Activation* RELUs for the nonlinear classification functions

**Layer 4:** Fully Connected. This has 120 outputs.

*Activation* RELUs for the nonlinear classification functions

**Layer 5:** Fully Connected (Logits). This has 43 outputs.

**Optimizer Types**

I tested the different options for the optimizers. In addition to the original AdamOptimizer, I tried to use GradientDescentOptimizer, RMSPropOptimizer, ProximalGradientDescentOptimizer, MomentumOptimizer. Looking at the rate of convergence of the model, only the RMSPropOptimizer came close to the AdamOptimizer. So, I continued to use the AdamOptimizer. The results are shown below. (These are numbers from the old submission)

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
No of Epochs = 64
BatchSize = 32
KernelSize = 5x5
Lenet:mu = 0.0
Lenet:sigma = 0.01
Learning_rate = 0.001


| Layer         		    |     Description	        			|
|:---------------------:|:---------------------------------------------:|
| Input         		    | 32x32x1 Gray image   					|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 32x32x64 	|
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


To train the model, I used the AdamOptimizer for the type of optimizer. I also used an Epoch size of 20, a batch size of 64 and a convolution KernelSize of 5x5. I kept the learning_rate at 0.001. I also shuffled the training images and the labels (simultaneously) to prevent any correlation to the order in the inputs.

My final model results were:
* training set accuracy of 0.93
* validation set accuracy of 0.93
* test set accuracy of 100

I did not use an iterative approach as I was short on time.

I chose the given LeNet solution architecture which recognizes numbers because it was based on an input image and seemed to perform well with the classification the input image into 10 numbers.
The steady rise in the accuracy of the model during a typical training from around 0.28 to a final 0.948 in 64 epochs gives me confidence that the neural network is learning to classify the test data into the 43 traffic signs.
I tried to speed up the learning by increasing the learning rate from 0.001 to 0.01 and the result was not good.

![alt text][image15]

**Testing the Model on New Images**

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6]
![alt text][image7] ![alt text][image8]

The first thing to notice about these images are their sizes are very different from the standard input image size of 32x32x3 used in for testing and validation of the neural network. So I read in each of these images and used the Python Imaging Library (PIL) to resize the images to the standard input size.

The hardest signs are the SpeedLimit signs which all are similar in appearance except for a digit inside. They all have the same circular shape and color. This sign was the one which scored least below hundred in the probability for the first guess at 95%. The RightOfWay sign also scored below 100 probably because there are several signs with the triangle shape. The other three signs seemed to get their first guess probability at 100%.

Here are the results of the prediction:

| Image			            |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 3.5Ton Limit					| 3.5Ton Limit 									    | 100%
| SpeedLimit 30      		| SpeedLimit 30   									| 95%
| KeepRight       			| KeepRight            							| 100%
| Turn-Right-Ahead   		| Turn-Right-Ahead					 				| 100%
| Right-of-way    			| Right-of-way										  | 98%


The model was able to correctly guess all 5 of the traffic signs correctly, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 100%

The code for making predictions on my final model is located in the 16th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a 3.5 Ton Weight Limit because of the first guess probability is close to 1.0. The top three soft max probabilities for the rest of the five images are as follows.

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.00000000e+00        | 3.5Ton Limit  									  |
| 1.93463645e-08     		| NoPassing 										    |
| 7.23138008e-11				| SpeedLimit60    								  |

For the second image the corresponding 3 guesses were like so;
| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 9.54341769e-01        | SpeedLimit30  									  |
| 3.39025594e-02     		| SpeedLimit50									    |
| 7.54935900e-03				| SpeedLimit80							        |

For the third image the corresponding 3 guesses were like so;
| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 9.99999881e-01        | KeepRight  				    					  |
| 1.22650476e-07      	| TurnLeftAhead    						      |
| 2.24875478e-08				| Yield    							            |

For the third image the corresponding 3 guesses were like so;
| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 9.97610688e-01        | KeepRight  				    					  |
| 1.36939518e-03      	| Stop  						                |
| 4.46832477e-04				| StraightOrRight     						  |

For the fifth image the corresponding 3 guesses were like so;
| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 9.78373647e-01        | Right-of-way  				    			  |
| 1.64140761e-02      	| DoubleCurve   			      		    |
| 3.12424079e-03				| BewareOfIceAndSnow							  |

*Classification of the Five Signs*

I displayed the final results of the classification of the five signs as shown below. The first image shows the example image. The next 3 images in the row show the results of the classifier arranged in the decreasing order of probablility. The probablility computed is displayed above the signs. The classifier successfully classified all five signs consistently over multiple tests.

![alt text][image17]

**Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)**

I used the given and tensorflow functions to display the hidden layers of the CNN graphically as shown below.
The original sign fed to the CNN is the 3.5TonLimit sign which is shown below.

![alt text][image14]

The first layer of hidden networks shows that the circular pattern that would be a good classifying feature is detected here in several of the convolution weight images. This is a good indicator that the hidden layers are doing the right thing by finding the appropriate features for classification automatically.
![alt text][image11]
The second layer of six images is shown below.
![alt text][image12]
The third layer of 16 images is shown below.
![alt text][image13]
