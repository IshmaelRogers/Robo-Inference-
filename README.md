# Robo-Inference-

free datasets

Kaggle https://www.kaggle.com/datasets
UCI: https://archive.ics.uci.edu/ml/datasets.html
[MNIST](http://yann.lecun.com/exdb/mnist/)

[image1]: inference_overview
[image2]: training_graph
[image3]: inference 
[image4]: training_graph2
[image5]: newClassModel

Inference development 
--

# 1. Introduction/Overview 

Inferencing involves a robotic system that can recognize objects from a live camera feed. To accomplish this task, Deep Learning techniques are used design networks capable of "learning" about the world around it. Training a network is an iterative process and it requies:

DIGITS - Offer a workflow that allow us to visualize and rapidly prototype robotic systems using inferencing. 


1. Managing data 
2. Designing a network 
3. Training the network
4. Mointor performance in real time and select the best performing model for deploying 
5. Comes preloaded with award winning networks for image classification and other tasks


1. Gathering example data
2. Designing an optimal network 

![alt text][image3]

NOTE: Networks are untrained artificial neural networks and models are what networks become once they exposed to data and trained. 


Training
-- 
![alt text][image1]

The process of training data involves the seperation of data in 3 sets:

0. Training set the previous data have been giving proper labels 
1. Validation set is used to prevent the classifier from learning directly from the test set. 
2. Testing set (if the data is sufficently large, take one or more subsets of test data and do not perform any analysis on them. 

NOTE: Overfitting should be avoided at all costs. Please see the Deep Learning REPO for more details 


Inferenceing is the process of using a live camera feed to classify real data by a network in real time. To do this we must train previous data using multiple models with different paramters.


NOTE: Image classification is the task of identifying the *predominant* object in an image. In the section we are using images that only contain *one* object. 

The next section is an example using the MINST data set using the DIGITS workflow.


# 2. DIGITS MINST Example

To train a network we need a lot of images that are sorted by their classified labels. In the case of handwritten numbers we will put all zero images in a directory named zero, and so on.

Should be nested under a single directory

path ``data/train_small``

The goal is to make the training and validation loss go down. If they diverge the model has overtrained and the epoch should be lowered. 


# 3. Lab: Nvidia DIGITS 

PART 1: Load, Organize and Train the Data 
---

1. Select Datasets tab on the left hand side
2. Select Classification from *Images drop down menu
3. Copy and paste the following filepath into the field "Training Images":  <code> /data/mnist_numbers/train_small </code>
4. Name the dataset so that it can be found. We've chosen: <code>Default Options Small Digits Dataset</code>
5. Click create. From here DIGITS will create a dataset from the folder.The *train_small folder contains 10 subfolders one for each class (Handwritten numnbers from zero to 9)
6. Explore the data by selecting "Explore the db" 
7. Navigate to the tab where DIGITS is still open and retunr to the main screen by clicking DIGITS on the top left of the screen. 
8. Choose the dataset we just created. (Default Options Small Digits Dataset)
9. Set epoch to 5 (epoch is one trip through the entire training datase. It is how long we want to train the dataset)
10. Select the AlexNet network (we are using 256x256 color images) 
11. Name the model 
12. Press create (the model is now training)



Results 
--
1. Training graph below shows how a network was trained in a 5 minute time frame to produce a model that can map images of handwritten digits to the number they represent with an accuracy of about 87%.

The blue curve shows how far off each prediction was from the actual label 
The green curve shows the difference between the model's predictions and actual labels for new data that it has learned from
The orange curve is the inverse of that loss (the difference between predicted and actual labels)

![alt text][image2]


PART 2: Testing trained networks 
---
Classify a single instance of new data
1. Type in this file path --> /data/mnist_numbers/train_small
2. Name dataset so it can be found later. our name --> Default Options Small Digits Dataset
3. Press Create. DIGITS is now creating a data se

PART 3: Improving the Model 
--
1. Click DIGITS logo on top left 
2. Select the models. Digits will then display the training graph generated while the model was training. The following three quantities are reported. 

1. Training loss should decrease from epoch to epoch 
2. Validation loss should decrease from epoch to epoch 
3. Accuracy is the measure of the ability of the model to correctly classify validation data. 

PART 4: STUDY MORE
---
Improve the accruary by making the model study more,

Exploring epochs
---
epochs
---

An epoch can be compared to one trip through a deck of flashcards. The following example illustrates what is happening during each epoch.

How a robot learns:

1. Neural networks take the first image (or small group of images) and make a prediction about what it is (or they are).
2. Their prediction is compared to the actual label of the image(s).
3. The network uses information about the difference between their prediction and the actual label to adjust itself.
4. The network then takes the next image (or group of images) and make another prediction.
5. This new (hopefully closer) prediction is compared to the actual label of the image(s).
6. The network uses information about the difference between this new prediction and the actual label to adjust again.
7. This happens repeatedly until the network has looked at each image.

How a human learns:

1. A student looks at a first flashcard and makes a guess about what is on the other side.
2. They check the other side to see how close they were.
3. They adjust their understanding based on this new information.
4. The student then looks at the next card and makes another prediction.
5. This new (hopefully closer) prediction is compared to the answer on the back of the card.
6. The student uses information about the difference between this new prediction and the right answer to adjust again.
7. This happens repeatedly until the student has tried each flashcard once.

Instructions:
--

1. scroll to bottom of model page and click the "Make Pretrained Model" button. This will save two things
  a. The network architeture that we selected
  b. What the model has learned based on the parameters that have been adjusted as the network trained during the 5 epochs.
  NOTE: We can now create a mnodel from this starting point.
2. GO to DIGITS home screen create a new image classification model
  a. Select the same dataset - (Default Options Small Dataset)
  b. Choose some number of epochs between 3 and 8. (Note that in creating a model from scratch, this is where you could have       requested more epochs originally.)
  c. This time, instead of choosing a "Standard Network," select "Pretrained Networks."
  d. Select the pretrained model that you just created, likely "My first model."
  e. Name your model - We chose "Study more"
  f. Click Create
  
  ![alt text][image4]
  
  NOTE: 
  1. The accuracy starts close to where our first model left off, 86%
  2. Accuracy continues to increase 
  3. The Rate of increase in accuracy slows down (more trips through data isn't the only way to increase performance)
  
PART 5: Test the new model
---

image path --> /data/mnist_numbers/test_small/2/img_4415.png

  Test with the following image paths 
  
  NOTE: The first number in the image filename is the digit in the image
  
  1. image-1-1.jpg

  2. image-2-1.jpg

  3. image-3-1.jpg

  4. image-4-1.jpg

  5. image-7-1.jpg

  6. image-8-1.jpg

  7. image-8-2.jpg

PART 6: Classify Multiple Objects 
---

Classify multiple files by putting them in lists

1. Navigate to right side of the DIGITS model page "test a list of images"
2. Press the Browse button and select an_image.list
3. Press classify 

Be sure to read the documentation of the MNIST dataset


# Matching Data to the right model 

1. Load 28x28 grayscale images and pick a model that is built to accept that type of data, LeNet. 
2. Click *explore the db
3. Create a model using the settings below 
![alt text][image5]

Results
--
The model improved performance. This one was accurate to more than 96%.
The model trained faster. In far less than two minutes, the model ran through 8 epochs.

NOTE: We can train faster and also experiment more

PART 7: Train with more data
---
Now we'll use the full MINST dataset instead of the 10% like before. We'll use the clone option in DIGITS to create a new job with similar properties to an older job.

1. Click DIGITS in upper left corner 
2. Click Dataset from the left side of the page to see previously created datasets. From there you will see *MNIST small* dataset.
3. Press Clone Job 
4. Create a database with the full MNIST data by changing the following settings:
  * Training Images - /data/mnist_numbers/train_full
  * Dataset Name - MNIST full
5. Press create 
6. Clone training model repeat above steps and change these values
  * Select the MNIST full dataset
  * Change the name to MNIST full
7. Create the model 

PART 8: Improving model result - Data Augmentation
--

You can see with our seven test images that the backgrounds are not uniform. In addition, most of the backgrounds are light in color whereas our training data all have black backgrounds. We saw that increasing the amount of data did help for classifying the handwritten characters, so what if we include more data that tries to address the contrast differences?

Augment the data by inverting the original images i.e turn the white pixels to black and vice-versa. Then we will train our network using the original and inverted images and see if classification is improved.

1. Clone and create a new dataset and model using the instructions above. 
  Augmented data directories - Training Images - /data/mnist_numbers/train_invert
2. Change the name of the dataset and model
3. Explore the database 
4. Train the model by cloneing the previous model results and change the dataset to the one created with the inverted images. 
5. Change the name of the model and create a new model

The augmented dataset helped better classify our images because it increase out overall datasize. 

Part 9: Modifying the network
---

It is possible to create custom networks to modify the existing ones, use different networks from external sources, or create one from scratch.

Follow these steps to modify a network

1. Select the Customize link on the right side of the Network dialog box
2. Examine code 
3. Click visiualize to see all of the layers of the model and how they are connected. 
4. Connect a ReLU to the first pool 
5. Change the values of num_output to 75 for *Conv1 and 100 for *Conv2
6. Visualize the model 
7. Change the name of the model and press the *Create* button  

The ReLU layer should go below the pool1 definition like so:

``` 
layer { 
  name: "reluP1"
  type: "ReLU"
  bottom: "pool1"
  top: "pool1"
 }
```

The Convolution layers should be changed as following:

```

layer {
  name: "conv1"
  type: "Convolution"
  bottom: "scaled"
  top: "conv1"
...
  convolution_param {
    num_output: **75**
...
 
layer {
  name: "conv2"
  type: "Convolution"
  bottom: "pool1"
  top: "conv2"
...
  convolution_param {
    num_output: **100**
...

```


The network is defined as a series of layers. Each layer is named so that its function is described. 

Layer characteristics
--
1. Top and bottom
2. Type defines what type the layer is
  * Convolution
  * Pool 
  * ReLU



# 4. Inference on the Jetson 

  1. Make sure git and cmke are installed locally 
  
  ``` $ sudo apt-get install git cmake ```
  2. clone the jetson-inference repo
  
  ``` $ git clone https://github.com/dusty-nv/jetson-inference ```
  3. Configuring with cmake - When cmake is run, a special pre-installation script ```CMakePreBuild.sh)``` is run and will automatically install any dependencies. 
  ```
  $ cd jetson-inference
  $ mkdir build
  $ cd build
  $ cmake ../
  ```
  4. Compile the project (make sure to be in the jetson-inference/build directory)
  
  ```
  $ make
  ```
  5. Classifying Images with ImageNet
  
  The imageNet object accepts an input image and outputs the probability for each class. 
  
  As examples of using imageNet we provide a command-line interface called:
  
  * imagenet-console https://github.com/dusty-nv/jetson-inference/blob/master/imagenet-console/imagenet-console.cpp
  * live camera program called imagenet-camera. https://github.com/dusty-nv/jetson-inference/blob/master/imagenet-camera/imagenet-camera.cpp
  
  6. Using the Console Program on Jetson
  
  1. Use the imagenet-console program to test imageNet recognition on some example images.
  
  * It loads an image
  * Uses TensorRT and the imageNet class to perform the inference
  * Overlays the classification 
  * Saves the output image
  
  2. Classify an example image with the imagenet-console program. 
  NOTE: imagenet-console accpets 3 command-line arguments
   * The path to the input image 
   * The path to the output image
   
  Example:
  
  ```$ ./imagenet-console orange_0.jpg output_0.jpg
  ```
  
  NOTE: After building make sure the terminal is located in the aarch64/bin directoy
  
  ``` $ cd jetson-inference/build/aarch64/bin 
  ```
  
  6. Classifying Live Video Feed from the Jetson onboard camera with imageNet.
    1. Navigate to real time image recognition demo in the */aarch64/bin called imagenet-camera
    2. Chose one of the following 
    
    ```
       $ ./imagenet-camera googlenet   # to run using googlenet
       $ ./imagenet-camera alexnet     # to run using alexnet   
    ```
  Note: By default the application can recognize up to 1000 different types of objects, since Googlenet and Alexnet are trained on the ILSVRC12 ImageNet database which contains 1000 classes of objects. The mapping of names for the 1000 types of objects, you can find included in the repo under https://classroom.udacity.com/nanodegrees/nd209/parts/dad7b7cc-9cce-4be4-876e-30935216c8fa/modules/4899a747-7c0d-4f40-9ab8-4f2eaf27a810/lessons/320f676e-0cae-4f44-80cc-5291cb2f673b/concepts/data/networks/ilsvrc12_synset_words.txt
  
 
  
  
  
  
  
  
  
  

# 5. Deploying models

# 6. Lab: Nvidia Development 








DIGITS WORKFLOW 

PROJECT IDEA 

DATA COLLECTION 

DOCUMENTATION 

DEPLOYING ON JETSON TX2

WORKSPACE 


RECAP 

PROJECT





