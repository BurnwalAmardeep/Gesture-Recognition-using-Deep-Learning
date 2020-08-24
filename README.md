# Gesture-Recognition-using-Deep-Learning
Gesture Recognition using Deep Learning

# Problem
Imagine you are working as a data scientist at a home electronics company which manufactures state of the art smart televisions. You want to develop a cool feature in the smart-TV that can recognise five different gestures performed by the user which will help users control the TV without using a remote. This solution will train a model to identify the gesture, which can then be used to perform certain actions like increasing/decreasing the volume of TV, changing the channel etc.


The gestures are continuously monitored by the webcam mounted on the TV. Each gesture corresponds to a specific command:
In this solution, below five gestures are tracked:
1. Thumbs up:  Increase the volume
2. Thumbs down: Decrease the volume
3. Left swipe: 'Jump' backwards 10 seconds
4. Right swipe: 'Jump' forward 10 seconds  
5. Stop: Pause the movie

# Understanding the Dataset
The required data can be downloaded from [google drive](https://drive.google.com/uc?id=1ehyrYBQ5rbQQe6yL4XbLWe3FMvuVUGiL).
The training data consists of a few hundred videos categorised into one of the five classes. Each video (typically 2-3 seconds long) is divided into a sequence of 30 frames(images). These videos have been recorded by various people performing one of the five gestures in front of a webcam - similar to what the smart TV will use. 
The data is in a zip file. The zip file contains a 'train' and a 'val' folder with two CSV files for the two folders. These folders are in turn divided into subfolders where each subfolder represents a video of a particular gesture. Each subfolder, i.e. a video, contains 30 frames (or images). Note that all images in a particular video subfolder have the same dimensions but different videos may have different dimensions. Specifically, videos have two types of dimensions - either 360x360 or 120x160 (depending on the webcam used to record the videos).
Each row of the CSV file represents one video and contains three main pieces of information - the name of the subfolder containing the 30 images of the video, the name of the gesture and the numeric label (between 0-4) of the video.
The data contains below four important files:
- train.csv : This file contains the foldername to be checked in train file, and the category of gesture action in numerical format separated by semicolon(;). This file is used in training purpose.
- train folder : This contains folders which contains the sequence of 30 frames(images) created from one video of a gesture. This is used in training the model.
- val.csv : This file contains the foldername to be checked in train file, and the category of gesture action in numerical format separated by semicolon(;). This file is used for validation purpose.
- val folder : This contains folders which contains the sequence of 30 frames(images) created from one video of a gesture. This is used in validation of trained model.

# Solution Approaches
In this we have used two solution architectures:
1. Conv2D + Recurrent Neural Network: The conv2D network will extract a feature vector for each image, and a sequence of these feature vectors is then fed to an RNN-based network. The output of the RNN is a regular softmax (for a classification problem such as this one).\
**Note:** There are a few key things to note about the conv-RNN architecture:
    - We can use transfer learning in the 2D CNN layer rather than training our own CNN(this helps in reducing the training time) 
    - GRU can be a better choice than an LSTM since it has lesser number of gates (and thus parameters)
2. Conv3D: 3D convolutions are a natural extension to the 2D convolutionsh. Just like in 2D conv, we move the filter in two directions (x and y), in 3D conv, we move the filter in three directions (x, y and z). In this case, the input to a 3D conv is a video (which is a sequence of 30 RGB images). If we assume that the shape of each image is 100x100x3, for example, the video becomes a 4-D tensor of shape 100x100x3x30 which can be written as (100x100x30)x3 where 3 is the number of channels. Hence, deriving the analogy from 2-D convolutions where a 2-D kernel/filter (a square filter) is represented as (fxf)xc where f is filter size and c is the number of channels, a 3-D kernel/filter (a 'cubic' filter) is represented as (fxfxf)xc (here c = 3 since the input images have three channels). This cubic filter will now '3D-convolve' on each of the three channels of the (100x100x30) tensor.

# Solution Files:
Below solution files are added:
