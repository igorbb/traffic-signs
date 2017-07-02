#**Traffic Sign Recognition**

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/bar.png "Histogram breakdown"
[image2]: ./examples/43k.png "Classes Samples"
[image3]: ./examples/aug.png "Augmented"
[image4]: ./examples/5img.png "5 Samples"
[image5]: ./examples/soft.png "Classifications"
[image6]: ./examples/gabor.png "Classifications"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  


###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used numpy to calculate basic stats of the traffic sign dataset.

* The size of training set is 34799 images/samples.
* The size of the validation set is 4410 images/samples.
* The size of test set is 12630 images/samples.
* There is a total of 51839 images/samples.
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set.
It is a bar chart showing the  normalized frequency per class/labels as well as train, validation and test sets.

![histogram][image1]

Here are a few samples of the 43 classes:

![visualization][image2]


###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

Given previous experience i decided to go directly to RGB images that were
normalized to be in the [0,1] interval.

To achieve the performance requirements of this task no data augmentation was necessary.
The provided dataset was enough to achieve high classification rates when trained with a sensible ANN.
Nevertheless, we implement data augmentation to further improve classification accuracy.
Data augmentation is done by randoml rotation images of the training set while training.

![visualization][image3]

The above image is showing the result of data augmentation. The first image shows the original
input and the second  shows a sample of data augmentation. The angle of rotation is randomly selected between [-20,20] degrees.


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.


For this task a few generic network functions were implemented:

**A generic `conv2d_maxpool` function.**
This function implements a convolution block followed by ReLu activation and a max pooling block.
Both convolution and max pooling blocks use `same padding`.
To document this function on a summary table assume:
* FM = Feature maps.
* CK = Convolution kernel.
* CS = Convolution stride.
* MK = Max Pooling kernel.
* MS = Max Pooling stride.

**A generic `fully_conn` function.**
This function applies a fully connected layer to any given layer.
It ReLu for activation.
TO document this function on a summary table assume:
* NO = Number of outputs neurons.


**A generic `output` function.**
Similar behavior to fully_conn. But no activation function is used.
The outputs are logits for the 43 classes.
In practice  the final layer is using softmax when we pass the logits to
`tf.nn.softmax_cross_entropy_with_logits`.


Finally we can summarize the network in a single table
The network summary is the following:

| Layer         		|     Description	        					|  output|
|:---------------------:|:---------------:|:------------------------------:|
| Input         	|        RGB image                 	| 32x32x3 							|
| conv2d_maxpool      	|FM= 32, CK=3x3, CS=1x1, MK=1x1, MS=1x1| 32x32x32 |
| conv2d_maxpool      	|FM= 64, CK=3x3, CS=1x1, MK=1x1, MS=1x1| 32x32x64|
| conv2d_maxpool      	|FM= 128, CK=3x3, CS=1x1, MK=2x2, MS=1x1| 32x32x128|
| flatten    	| Transform tensor to vector| 131072|
| fully_conn    	| NO=256 | 256|
| dropout    	| keep_probability=0.55 | 256|
| fully_conn    	| NO=256 | 256|
| dropout    	| keep_probability=0.55 | 256|
| output    	| softmax| 43|



####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model we setup the following hyperparameters for the Adam optimizer
* epochs = 60
* batch_size = 512
* keep_probability = 0.55
* lr = 0.001

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.


The first network architecture I tried was the LeNet we implemented previously.
Since the notebook already provided me with an expected performance, I used it has a sanity check.
This way I could assess if my training function and performance assessments were working.

The next step was to change the network for something for sensible.
Previous experience, made it easier to select a network architecture.

The selected architecture is the one presented in section 2.
Now the network allowed me to adjust dropout and the learning rate.
I have used those controls to ensure good regularization and
that the model was not overfitting.

The dropout was key in for the achieved results. With a keep_probability of 0.55
We enforce a strong regularization and stopped with overfitting issues.

My final model results were:
* training set accuracy of  100%
* validation set accuracy of  98%
* test set accuracy of 96.7%

The loss on validation seems to be not indicating overfitting, and the close range accuracy on both validation and testing are indications of a well trained model that is generalizing to new observations.

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4]
The first image is challenging due to a flare effect.
The second image is slanted which is also a challenge.
The fourth image has many signals in the background, which might be a challenge.
The remainder images have well placed signs.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

![results][image5]

The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%.
For our sample size, these are good results. As an addendum the original test has 97% accuracy.
But to have assessments with such granularity we would need a lot of images, much more than five.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 23rd cell of the Ipython notebook.

As we can see, we only have one  miss-classification.
This miss-classification was on a 20km/h speed limit sign.
But the network was not very sure on the classifications for that image.
The first guess was 30km/h speed limit sign, with probabilities around 0.65.
The the correct class was still the second most probable guess with 0.35 probability.

All the other images were correctly classified with very high confidence.

Nevertheless it seems that the network has a hard time differentiating  speed limits signs.
I assume that this is due to the very low resolutions of some of the text inside our speed signs (in training set).


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

The first layer of the neural network is working as a Gabor filter bank.
We can see that the outputs are likely very informative for the segmentation of signs.
This is clear on the attached images:

![gabor][image6]

The network is segmenting color, which  is very informative on for differentiating signs.
Basic color segmentation might be very informative for foreground/background segmentation.
We also have basic edge detectors which are fundamental for CNN that learns hierarchical representations of image concepts.
