# Robo-Inference-

free datasets

Kaggle https://www.kaggle.com/datasets
UCI: https://archive.ics.uci.edu/ml/datasets.html
[MNIST](http://yann.lecun.com/exdb/mnist/)

[image1]: ./IMAGES/inference_overview.png
[image2]: ./IMAGES/training_graph.png
[image3]: ./IMAGES/inference.png
[image4]: ./IMAGES/training_graph2.png
[image5]: ./IMAGES/newClassModel.png
[image6]: ./IMAGES/metricsTable.png
[image7]: ./IMAGES/speedvsaccuracy.png
[image8]: ./IMAGES/tx1vstx2.png
[image9]: ./IMAGES/sw_upgrades.png

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
6.1 From the home screen, the "Models" tab will be pre-selected. Click "Images" under "New Model" and select "Classification",
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

# Inferencing applications in robotics

1.Embedded Examples
--

Question: Why is inference speed and AI at the edge so important?

There are many use cases where systems cannot rely on clouds to solve problems. 

Example 1: AI Cities - Use Intelligent Video Analytics (IVA) to sense and make intelligent decisions with no physical interaction. They usually have large bandwidths the amount of data generated are vast. Therefore the network cannot handle all the data alone. 

Inferencing in IVA

Inference is used to analyze a live camera feed to provide many powerful insights about the city.

1. Parking spot location
2. Security Analysis 
3. Behavioral analysis


Example 2: Privacy - Data security is very important to some companies because the privacy of indiduvals need to be protected.



Latency is a time delay between an input event being applied to the system, and the associated output action from the system. In robotics there are devices that support satefy critical services and need to make millisecond decisions very quickly. 

Connectivity - For applications in remote areas where robots and UAV's are ideally suited, network coverage may be weak or non-existent or even slower using 2G or satellite communications. If connection is not always available, relying on a cloud is less than ideal.

Other Latency factors - Keeping track of how long it takes to react to incoming data is always a consideration in real-time systems.

1. Complexity of the Model
2. Available memory 
3. Computational loads



2 Metrics for Inference Deployment
---

1. Analysis of DNN's using additional computation metrics provide insight into design constraints in deployable systems that use DNN's for inference. Architertures can be compared using the following metrics

* Top1 Accuracy 
* Operations count
* Network Parameters Count
* Inference Time 
* Power Consumption 
* Memory Usage 

![alt text][image6] 

Key Insights for optimizing a deployable robotic system using inference from Canziani's paper:

* Power consumption is independent of batch size and architecture
“When full resources utilisation is reached, generally with larger batch sizes, all networks consume roughly an additional 11.8W”
* Accuracy and inference time are in a hyperbolic relationship
“a little increment in accuracy costs a lot of computational time”
* Energy constraint is an upper bound on the maximum achievable accuracy and model complexity
“if energy consumption is one of our concerns, for example for battery-powered devices, one can simply choose the slowest architecture which satisfies the application minimum requirements”
* The number of operations is a reliable estimate of the inference time.


3. Speed/accuracy Tradeoff

Two conclusions are helpful when considering the speed and accuracy tradeoff for a robotic system design that uses DNN inference on a deployed platform:

* Accuracy and inference time are in a hyperbolic relationship
* The number of operations is a reliable estimate of the inference time.

If the system is supposed to be able to classify images that are included in the 1000 classes from ImageNet, then all that is needed is to benchmark some of the inference times on the target system, and choose the network that best meets the constraints of inference time or accuracy required. 

Example: The following chart characterizes inference speed vs accuracy 
![alt text][image7]

The system requires an accuracy of at least 80%, the best fps we can hope for is about 10 fps as an upper bound. If that is acceptable, the system can be built using a known architecture trained on ImageNet

If 10 fps @ 80% accuracy isn't good enough, other options are available that may satisfy the contraints without redesigning a new network architecture:

* Redeploy to a higher performance hardware platform with improved operation execution time
* Upgrade the software platform with an optimized release that affects inference time
* Increase the accuracies by customizing the data to a smaller number of relevant classifications

Improved Hardware Platforms
---

![alt text][image8]


Improved Software Platform
---

For times where upgrading hardware is not an option, changes in software platforms that leverage optimizations can have the same ammount of impact.

![alt text][image9]

Customized Data
---

So far, the improvements discussed have all been explained in terms of the benchmarks using famous architectures such as AlexNet, GoogLeNet, and ResNet, running the ImageNet classification set of 1000 image classes. That’s just a benchmark, though. These same architectures can be used with alternate data sets, possibly with better accuracy. For example, the ImageNet data set includes 122 different dog breeds among its labels. If it only matters for an application that the image is a “dog”, not which breed of dog, these images could be trained as a single class, and all the inaccuracies that might occur between categories would no longer cause errors.

Perhaps the application really is only concerned with picking items off of a conveyor belt and there are only 20 categories, but speed is a very important metric. In that case, the 20 classes could be trained (using an appropriately robust user data set), and a smaller, faster architecture such as AlexNet might provide a sufficient accuracy metric with the smaller number of classes.

These are all considerations that can be taken into account to inform the design process for a deployable system. In the end though, experimentation is still necessary to determine actual results and best optimizations.

Project: Robotic Inference 
---

DIGITS WORKFLOW 

Steps:

1. Start DIGITS server by entering the command ```digits``` into a terminal 
2. From another terminal run ```print_connection.sh``` to get the link for the DIGITS GUI. Keep this script running to keep the workspace active if a network is being trained. 
3. Add dataset into DIGITS (/data/P1_data
4. Choose a training model 
5. Test the trained model by running the ``` evaluate``` in another terminal with the DIGITS server still running (only after the model is done being trained). The job id will be requested
6. 

``evaluate`` checks the inference speed of your model for a single imput averaged over ten attempts for 5 runs. 

PROJECT IDEA 
---
Preception, Decision Making, Action


Project guidelines:
In addition to training a network on the supplied data, you will also need to choose and train a network using your own collected data. At a minimum, it needs to be a classification network with at least 3 classes. However, if you would like to be more adventurous you can use more than 3 classes and even subclasses!

If you are looking for an extra challenge, you can create a detection network. It will require you to annotate your data in addition to collecting it. More information can be found in the next section on this process.

Its okay to use a sample idea below if you’re having a hard time deciding what to do!

Resources and ideas:

* Pill identifier with classes: (pill a, pill b, pill c, no pill)
* Defective item vs normal item with classes: (no item, defective item, normal item)
* Person vs no person with classes: (correct person, wrong person, no person)
* Location of robot part on a workbench.
* Insert your idea here!

DATA COLLECTION
---

Workspace home directory ```/home/workspaces/```

Check the directory size with the following command 

```
$ du -sh /home/workspace

```

1. Classification Network - requires 400 images per class. 
NOTE: Collect images in the same environment in which you will be conducting your inference. 
2. Detection Network - data needs to be annotated before uploading to DIGITS. More specifically, bounding boxes need to be placed around what we want the network to learn. [image annotation][https://en.wikipedia.org/wiki/List_of_manual_image_annotation_tools]

Collecting the images
----

Images can be collected using a webcam and a Python or C++ script, phones or the jetson. Below is a basic python script for collecting images from a webcam 

# Python Data Capture Script
NOTE: Setup the proper environment with Python 2.7 using cv2: ```conda instal -c confa-forge opencv=2.4 ```

``` python

import cv2

# Run this script from the same directory as your Data folder

# Grab your webcam on local machine
cap = cv2.VideoCapture(0)

# Give image a name type
name_type = 'Small_cat'

# Initialize photo count
number = 0

# Specify the name of the directory that has been premade and be sure that it's the name of your class
# Remember this directory name serves as your datas label for that particular class
set_dir = 'Cat'

print ("Photo capture enabled! Press esc to take photos!")

while True:
    # Read in single frame from webcam
    ret, frame = cap.read()

    # Use this line locally to display the current frame
    cv2.imshow('Color Picture', frame)

    # Use esc to take photos when you're ready
    if cv2.waitKey(1) & 0xFF == 27:

        # If you want them gray
        #gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        # If you want to resize the image
        # gray_resize = cv2.resize(gray,(360,360), interpolation = cv2.INTER_NEAREST)

        # Save the image
        cv2.imwrite('Data/' + set_dir + '/' + name_type + "_" + str(number) + ".png", frame)

        print ("Saving image number: " + str(number))

        number+=1

    # Press q to quit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

# C++ script using the Jetson

``` cpp

/*
Compile with:

gcc -std=c++11 Camera_Grab.cpp -o picture_grabber -L/usr/lib -lstdc++ -lopencv_core -lopencv_highgui -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs

Requires recompiling OpenCV with gstreamer plug in on. See: https://github.com/jetsonhacks/buildOpenCVTX2

Credit to Peter Moran for base code.
http://petermoran.org/csi-cameras-on-tx2/
*/


#include <opencv2/opencv.hpp>
#include <string>

using namespace cv;
using namespace std;

std::string get_tegra_pipeline(int width, int height, int fps) {
    return "nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)" + std::to_string(width) + ", height=(int)" +
           std::to_string(height) + ", format=(string)I420, framerate=(fraction)" + std::to_string(fps) +
           "/1 ! nvvidconv flip-method=0 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";
}

int main() {
    // Options
    int WIDTH = 500;
    int HEIGHT = 500;
    int FPS = 30;

    // Directory name
    string set_dir = "Test";
    // Image base name
    string name_type = "test";

    int count = 0;

    // Define the gstream pipeline
    std::string pipeline = get_tegra_pipeline(WIDTH, HEIGHT, FPS);
    std::cout << "Using pipeline: \n\t" << pipeline << "\n";

    // Create OpenCV capture object, ensure it works.
    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);
    if (!cap.isOpened()) {
        std::cout << "Connection failed";
        return -1;
    }

    // View video
    cv::Mat frame;
    while (1) {
        cap >> frame;  // Get a new frame from camera

        // Display frame
        imshow("Display window", frame);

        // Press the esc to take picture or hold it down to really take a lot!
        if (cv::waitKey(1) % 256 == 27){

            string string_num = to_string(count);

            cout << "Now saving: " << string_num << endl;

            string save_location = "./" + set_dir + "/" + name_type + "_" + string_num + ".png";

            cout << "Save location: " << save_location << endl;

            imwrite(save_location, frame );

            count+=1;

        }

        cv::waitKey(1);
    }
}

```


Documentation

NOTE: Use LaTeX to create the report

NOTE: All submissions have to be watermarked with first and last name. Images, tables, charts and graphs

Abstract: A abstract is meant to be a summary of all of the relevant points in your presented work. It is designed to present a high-level overview of the report, providing just enough detail to convey the necessary information. The abstract may often mention a one-sentence summary of the results. While the type of voice chosen for the paper (active or passive) may be up for debate, you should avoid the use of “I” and “me” in the report. It usually is kept to a length of 150 - 200 words.

Example: You should not write, “I present two different neural networks for classifying my data”. Instead, you should try to say, “ Two different neural networks are used for classification”.

Introduction: The introduction should provide some material regarding the history of the problem, why it is important and what is intended to be achieved. If there exists any previous attempts to solve this problem, this is a great place to note these while conveying the differences in your approach (if any). The intent is to provide enough information for the reader to understand why this problem is interesting and set up the conversation for the solution you have provided.

Use this space to introduce your robotic inference idea and how you wish to apply it. If you have any papers / sites you have referenced for your idea, please make sure to cite them.

Background / Formulation: At this stage, you should begin diving into the technical details of your approach by explaining to the reader how parameters were defined, what type of network was chosen, and the reasons these items were performed. This should be factual and authoritative, meaning you should not use language such as “I think this will work” or “Maybe a network with this architecture is better..”. Instead, focus on items similar to, ”A 3-layer network architecture was chosen with X, Y, and Z parameters”

Explain why you chose the network you did for the supplied data set and then why you chose the network used for your robotic inference project.

Data Acquisition: This section should discuss the data set. Items to include are the number of images, size of the images, the types of images (RGB, Grayscale, etc.), how these images were collected (including the method). Providing this information is critical if anyone would like to replicate your results. After all, the intent of reports such as these is to convey information and build upon ideas so you want to ensure others can validate your process.

Justifying why you gathered data in this way is a helpful point, but sometimes this may be omitted here if the problem has been stated clearly in the introduction. It is a great idea here to have at least one or two images showing what your data looks like for the reader to visualize.

Results: This is typically the hardest part of the report for many. You want to convey your results in an unbiased fashion. If your results are good, you can objectively note this. Similarly, you may do this if they are bad as well. You do not want to justify your results here with discussion; this is a topic for the next session. Present the results of your robotics project model and the model you used for the supplied data with the appropriate accuracy and inference time.

For demonstrating your results, it is incredibly useful to have some charts, tables, and/or graphs for the reader to review. This makes ingesting the information quicker and easier.

Discussion: This is the only section of the report where you may include your opinion. However, make sure your opinion is based on facts. If your results are poor, make mention of what may be the underlying issues. If the results are good, why do you think this is the case? Again, avoid writing in the first person (i.e. Do not use words like “I” or “me”). If you really find yourself struggling to avoid the word “I” or “me”; sometimes, this can be avoided with the use of the word “one”. As an example: instead of, "I think the accuracy on my dataset is low because the images are too small to show the necessary detail" try, "one may believe the accuracy on the dataset is low because the images are too small to show the necessary detail". They say the same thing, but the second avoids the first person.

Reflect on which is more important, inference time or accuracy, in regards to your robotic inference project.

Conclusion / Future Work: This section is intended to summarize your report. Your summary should include a recap of the results, did this project achieve what you attempted, and is this a commercially viable product? For future work, address areas of work that you may not have addressed in your report as possible next steps. This could be due to time constraints, lack of currently developed methods / technology, and areas of application outside of your current implementation. Again, avoid the use of the first-person.


DEPLOYING ON JETSON TX2

1. Download the model from DIGITS to the jetson
  * Navifate in a browser on the Jetson to the DIGITS server. 
  * Download the model 
2. Create a folder on the system with a name for the model and extract the contents of the downloaded file into that folder with the ```tar -xzvf ``` command. 
3. Create an environment variable called NET to the location of the model path
  ``` export NET=/home/user/Desktop/my_model ``` into the terminal
 4. Navigate to the Jetson inference folder then into the executable binaries and launch imagenet or detect net like so
   ```
   ./imagenet-camera --prototxt=$NET/deploy.prototxt --model=$NET/your_model_name.caffemodel --labels=$NET/labels.txt --input_blob=data --output_blob=softmax
   ```
   
   
   #SPECIAL TOPIC: Actuation based on classifier data using imagenet or detectnet 
   
   To actuate based on the information from a classifier modify the C++ file accordingly!
   
   Here is an example of calling a servo action based on a classification result. It can be inserted into the code here at 
   line 168:
   
   ```
   std::string class_str(net->GetClassDesc(img_class));

if("Bottle" == class_str){
          cout << "Bottle" << endl;
          // Invoke servo action
}

else if("Candy_Box" == class_str){
         cout << "Candy_Box" << endl;
         // Do not invoke servo action
}

else {
         // Catch anything else here
}

```
   
WORKSPACE

See Udacity Nanodegree until I configure for the Jetson TX 2

PROJECT STEPS

1) Using the supplied data in the digits image, create a network that achieves at least 75 percent accuracy and an inference time of less than 10 ms. This can be done by running the evaluate command. Be sure to take a screenshot of the results and add the date and your name to the image.

2) Determine what you want your robotic inference idea to be. Collect your own data that fits with what your model is trying to accomplish. Train a network that produces meaningful results.

3) Document the project following the provided structure. Be sure to include and talk about both networks you worked on. You could use this [template][https://www.overleaf.com/read/ghypqqdcrjsv]

When you have completed the steps above and have cross checked them to the rubric here, go ahead and submit your work!

Be sure to include the following items in your project submission: 

A write up in PDF format that addresses the rubric points. Be sure to include sample images of the items you are classifying, charts, graphs, etc.
A photo of your terminal output after running the evaluate command with your name and the date watermarked on it. This can also just be included in your write up.
The model for the supplied data set. This includes: deploy.prototxt, labels.txt, mean.binaryproto, your_model.caffemodel, solver.prototxt and train_val.prototxt.


PROJECT RUBRIC 

Basic Requirements
---
* your write up (PDF format)
* any supporting images (including sample images of the items you are classifying, charts, graphs, etc.)
* your trained model for the supplied data set.


For the submission of your trained models include the following files in separate folders:

* deploy.prototxt
* labels.txt
* mean.binaryproto
* your_model.caffemodel
* solver.prototxt
* train_val.prototxt.

Also included, a photo of the evaluate command output as a screenshot with your name annotated on it.

watermarking tool [https://www.watermarquee.com]

Numerical Requirements 
---

* A network of your choice must be trained on the supplied data set and must fall below the required inference time of 10 ms on the supplied workspace.
* A network of your choice must be trained on the supplied data set and must surpass the required accuracy of 75%.

Write up requirements
---

Include a full write up with the following sections:

* Abstract
* Introduction
* Background / Formulation
* Data Acquisition
* Results
* Discussion
* Future Work.

Include supporting images where appropriate. All images, charts, tables, etc. must be watermarked in the report.


