# Table of Contents
* [Abstract](#abstract) 
* [Data Collection and Balancing the Dataset](#data-collection-and-balancing-the-dataset)
* [Data Augmentation and Preprocessing](#data-augmentation-and-preprocessing)
* [Neural Network Architecture and Training Process](#neural-network-architecture-and-training-process)
* [Putting Them All Together: Autonomous Driving](#putting-them-all-together-autonomous-driving)
* [File Descriptions and Usage](#file-descriptions-and-usage)
* [Demonstration Videos](#demonstration-videos)
* [Reference Paper](#reference-paper)

# Abstract 
In this project, self driving car simulation is carried out by reproducing a human performed driving pattern using the supervised learning based behavioral cloning approach. The required data are collected by driving couple of laps around Track #1 of the Udacity's self driving car simulator, various augmentation techniques are used to make the trained model generalize well enough so that the car drives autonomously on both Track #1 and an unseen track, Track #2, and making use of a convolutional neural network (CNN) based on the NVIDIA architecture, steering angles corresponding to the positions of the car on the track are learned. A real-time web app is used make the model and the simulator communicate continuously so that the essential information for autonomous driving such as the current position of the car, predicted steering angle and the throttle to be given are passed along them.   

# Data Collection and Balancing the Dataset
I drove around the Track #1 approximately 15 minutes to collect the images with the three cameras mounted on the left, center and right side of the car. Figure 1 shows a set of sample images.  

![Figure-1](https://user-images.githubusercontent.com/43919074/229631030-89cfca67-6fe9-4148-9e2d-175c30914bf7.png)
**Figure 1: Sample images taken from left, center and right side cameras**
 
Since Track #1 has lots of straights, there is a strong bias on the zero degree steering angle as the following histogram suggests.  

![Figure-2](https://user-images.githubusercontent.com/43919074/229843435-cf744305-a599-449b-8915-5d3ae4a676fe.png)  
**Figure 2: Distribution of the steering angles of the original dataset**

The neural network model would not be able to learn the sharp turns if this bias problem was not fixed. I overcame this issue by truncating the number of samples to 800 for each steering angle and ended up with the distribution shown in Figure 3.  

![Figure-3](https://user-images.githubusercontent.com/43919074/229843481-3cbba3f6-08c5-471b-af1a-a13fe051cd48.png)  
**Figure 3: Distribution of the steering angles of the balanced dataset**  

After balancing the dataset, the images split between training and validation sets with the ratios 80% and 20% respectively. 

# Data Augmentation and Preprocessing
In order to prevent overfitting and make the model operate on several tracks that have diverse characteristics (such as shape, surrounding environment, lighting condition etc.), augmenting the images in the training dataset is necessary. The following figure illustrates the augmented images with different techniques.  

![augmented-images](https://user-images.githubusercontent.com/43919074/229873377-04ba8a66-22ff-4c97-bc90-6a5d4d81668a.png)  
**Figure 4: (From top left to bottom right) zoomed-in, translated, darkened and horizontally flipped images**

Before training the model one last step, which is preprocessing, is required to improve the efficiency and speed-up the training process.  
First off, the image is cropped such that only the region of interest is left, then as proposed in the NVIDIA paper, the color model is changed to YUV and the width and height are set to 200 pixels and 66 pixels respectively, after that Gaussian blur is applied to reduce the noise, and finally normalization is used to work with pixel values between 0 and 1.   
Figure 5 shows an image after the mentioned preprocessing techniques are applied.  

![preprocessed-image](https://user-images.githubusercontent.com/43919074/229897130-3d9471d4-e099-4209-8073-d4353b23959d.png)  
**Figure 5: Preprocessed image**

# Neural Network Architecture and Training Process
I used a slightly modified version of the convolution neural network architecture proposed in the NVIDIA paper. Figure 6 indicates the layers, output shapes and the number of parameters of the model. Note that, I preferred to use ELU activation function since it can produce negative outputs as opposed to ReLU activation function. For the compilation of the model, Adam optimizer with learning rate 1e-4 is used and mean squared error loss function is chosen as it's a regression task.   

![Figure-6](https://user-images.githubusercontent.com/43919074/230517752-a88fee2d-61d0-48a6-9583-321933bcc073.png)  
**Figure 6: CNN model summary**
  
During the training process, the provided images to the CNN model are generated with batch generation on the fly so that they are not stored in the memory. Notice that, 384,000 images are generated for training and 256,000 images are generated for validation and these would be pretty huge number of images to fit in the memory in the absense of this method. Also, to be able to use significantly higher number of images than the size of the training and validation sets is another advantage of using a batch generator. In order to decide the best model, validation losses are considered. Validation losses after each epoch can be seen in Figure 7. 

![Figure-7](https://user-images.githubusercontent.com/43919074/230692258-54c22c4d-4bd7-4881-a7d2-770a955a56dd.png)  
**Figure 7: Training process**

# Putting Them All Together: Autonomous Driving
In order to make the car drive autonomously, bidirectional communication between the model and the simulator need to be established. To achieve this, a real time web application is built using Socket.IO and Flask. Using the real time web app, the simulator continuously sends the current frame recorded by the center camera, the position of the car on the track is used as an input to the model and the model predicts a steering angle. A throttle value is calculated based on the current and desired speed using a PI controller. The predicted steering angle and the calculated throttle value are sent back to the simulator and the car moves on the track accordingly.      

# File Descriptions and Usage
`drive.py` contains the class, function and event handlers to drive the car autonomously   
`utils.py` provides data manipulation functions and batch generator  
`model.py` is used to create and train the model  
`model.h5` contains the weights and model configuration

Follow the below steps to use these files:  
* Download [Udacity's self driving car simulator](https://github.com/udacity/self-driving-car-sim)
* Download [Anaconda](https://www.anaconda.com/products/distribution)  
* Launch the Anaconda prompt 
* Run the following commands to create an environment and install the necessary packages:   
~~~
conda create --name <ENVIRONMENT_NAME>
For Windows: activate <ENVIRONMENT_NAME> / For Linux and macOS: source activate <ENVIRONMENT_NAME>   
conda install -c anaconda flask
conda install -c conda-forge python-socketio
conda install -c conda-forge eventlet
conda install -c conda-forge python-engineio=3.0.0
conda install -c conda-forge tensorflow
conda install -c conda-forge keras 
conda install -c anaconda pillow
conda install -c anaconda numpy
conda install -c conda-forge opencv
conda install -c anaconda pandas
conda install -c anaconda scikit-learn
conda install -c conda-forge imgaug
~~~

If you want to train your own model:  
* You can either use my dataset or get your own dataset (launch the Udacity's self driving car simulator, click `Play!`, choose a track, select `TRAINING MODE` and click the `RECORD` button)     
* Make the changes you want on model.py and utils.py  
* Run model.py  
~~~ 
python model.py
~~~

To directly use my model or your own model.h5 that you'll have after running model.py:  
* Run drive.py   
~~~ 
python drive.py
~~~
* Launch the Udacity's self driving car simulator, click `Play!`, choose a track and select `AUTONOMOUS MODE`  

# Demonstration Videos
* Track #1: https://www.youtube.com/watch?v=O2iRXwEec4c  
* Track #2: https://www.youtube.com/watch?v=B_ts9OQ-NP0  

# Reference Paper
https://developer.nvidia.com/blog/deep-learning-self-driving-cars/    
