# Robo-Inference-

free datasets

Kaggle https://www.kaggle.com/datasets
UCI: https://archive.ics.uci.edu/ml/datasets.html
[MNIST](http://yann.lecun.com/exdb/mnist/)

[image1] inference_overview
[image2] training_graph
[image3] inference 

Inference development 
--

1. Introduction/Overview 

Inferencing involves a robotic system that can recognize objects from a live camera feed. To accomplish this task, Deep Learning techniques are used design networks capable of "learning" about the world around it. Training a network is an iterative process and it requies:

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


2. DIGITS MINST Example


Lab: Nvidia DIGITS 

# PART 1: Load, Organize and Train the Data 


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

epochs
---
The following two example illustrate what is happening during each epoch.


1. Neural networks take the first image (or small group of images) and make a prediction about what it is (or they are).
2. Their prediction is compared to the actual label of the image(s).
3. The network uses information about the difference between their prediction and the actual label to adjust itself.
4. The network then takes the next image (or group of images) and make another prediction.
5. This new (hopefully closer) prediction is compared to the actual label of the image(s).
6. The network uses information about the difference between this new prediction and the actual label to adjust again.
7. This happens repeatedly until the network has looked at each image.

Results 
--
1. Training graph below shows how a network was trained in a 5 minute time frame to produce a model that can map images of handwritten digits to the number they represent with an accuracy of about 87%.

![alt text][image2]


# PART 2: Testing trained networks 

Classify a single instance of new data
1. Type in this file path --> /data/mnist_numbers/train_small
2. Name dataset so it can be found later. our name --> Default Options Small Digits Dataset
3. Press Create. DIGITS is now creating a data se

# PART 3: Improving the Model 

1. Click DIGITS logo on top left 
2. Select the models. Digits will then display the training graph generated while the model was training. The following three quantities are reported. 

1. Training loss should decrease from epoch to epoch 
2. Validation loss should decrease from epoch to epoch 
3. Accuracy is the measure of the ability of the model to correctly classify validation data. 

# PART 4: STUDY MORE

Improve the accruary by making the model study more,




Methods for accessing data from the dataset
---
















Inference on the Jetson 

Deploying models

Lab: Nvidia Development 


DIGITS - The DIGITS offer tools that allow us to visualize and rapidly prototype robotic systems using inferencing. 


1. Managing data 
2. Designing a network 
3. Training the network
4. Mointor performance in real time and select the best performing model for deploying 
5. Comes preloaded with award winning networks for image classification and other tasks






DIGITS WORKFLOW 

PROJECT IDEA 

DATA COLLECTION 

DOCUMENTATION 

DEPLOYING ON JETSON TX2

WORKSPACE 


RECAP 

PROJECT





