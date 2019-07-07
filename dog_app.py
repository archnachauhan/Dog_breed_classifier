#!/usr/bin/env python
# coding: utf-8

# # Convolutional Neural Networks
# 
# ## Project: Write an Algorithm for a Dog Identification App 
# 
# ---
# 
# In this notebook, some template code has already been provided for you, and you will need to implement additional functionality to successfully complete this project. You will not need to modify the included code beyond what is requested. Sections that begin with **'(IMPLEMENTATION)'** in the header indicate that the following block of code will require additional functionality which you must provide. Instructions will be provided for each section, and the specifics of the implementation are marked in the code block with a 'TODO' statement. Please be sure to read the instructions carefully! 
# 
# > **Note**: Once you have completed all of the code implementations, you need to finalize your work by exporting the Jupyter Notebook as an HTML document. Before exporting the notebook to html, all of the code cells need to have been run so that reviewers can see the final implementation and output. You can then export the notebook by using the menu above and navigating to **File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.
# 
# In addition to implementing code, there will be questions that you must answer which relate to the project and your implementation. Each section where you will answer a question is preceded by a **'Question X'** header. Carefully read each question and provide thorough answers in the following text boxes that begin with **'Answer:'**. Your project submission will be evaluated based on your answers to each of the questions and the implementation you provide.
# 
# >**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut.  Markdown cells can be edited by double-clicking the cell to enter edit mode.
# 
# The rubric contains _optional_ "Stand Out Suggestions" for enhancing the project beyond the minimum requirements. If you decide to pursue the "Stand Out Suggestions", you should include the code in this Jupyter notebook.
# 
# 
# 
# ---
# ### Why We're Here 
# 
# In this notebook, you will make the first steps towards developing an algorithm that could be used as part of a mobile or web app.  At the end of this project, your code will accept any user-supplied image as input.  If a dog is detected in the image, it will provide an estimate of the dog's breed.  If a human is detected, it will provide an estimate of the dog breed that is most resembling.  The image below displays potential sample output of your finished project (... but we expect that each student's algorithm will behave differently!). 
# 
# ![Sample Dog Output](images/sample_dog_output.png)
# 
# In this real-world setting, you will need to piece together a series of models to perform different tasks; for instance, the algorithm that detects humans in an image will be different from the CNN that infers dog breed.  There are many points of possible failure, and no perfect algorithm exists.  Your imperfect solution will nonetheless create a fun user experience!
# 
# ### The Road Ahead
# 
# We break the notebook into separate steps.  Feel free to use the links below to navigate the notebook.
# 
# * [Step 0](#step0): Import Datasets
# * [Step 1](#step1): Detect Humans
# * [Step 2](#step2): Detect Dogs
# * [Step 3](#step3): Create a CNN to Classify Dog Breeds (from Scratch)
# * [Step 4](#step4): Create a CNN to Classify Dog Breeds (using Transfer Learning)
# * [Step 5](#step5): Write your Algorithm
# * [Step 6](#step6): Test Your Algorithm
# 
# ---
# <a id='step0'></a>
# ## Step 0: Import Datasets
# 
# Make sure that you've downloaded the required human and dog datasets:
# 
# **Note: if you are using the Udacity workspace, you *DO NOT* need to re-download these - they can be found in the `/data` folder as noted in the cell below.**
# 
# * Download the [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip).  Unzip the folder and place it in this project's home directory, at the location `/dog_images`. 
# 
# * Download the [human dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip).  Unzip the folder and place it in the home directory, at location `/lfw`.  
# 
# *Note: If you are using a Windows machine, you are encouraged to use [7zip](http://www.7-zip.org/) to extract the folder.*
# 
# In the code cell below, we save the file paths for both the human (LFW) dataset and dog dataset in the numpy arrays `human_files` and `dog_files`.

# In[2]:


import numpy as np
from glob import glob

# load filenames for human and dog images
human_files = np.array(glob("/data/lfw/*/*"))
dog_files = np.array(glob("/data/dog_images/*/*/*"))

# print number of images in each dataset
print('There are %d total human images.' % len(human_files))
print('There are %d total dog images.' % len(dog_files))


# <a id='step1'></a>
# ## Step 1: Detect Humans
# 
# In this section, we use OpenCV's implementation of [Haar feature-based cascade classifiers](http://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html) to detect human faces in images.  
# 
# OpenCV provides many pre-trained face detectors, stored as XML files on [github](https://github.com/opencv/opencv/tree/master/data/haarcascades).  We have downloaded one of these detectors and stored it in the `haarcascades` directory.  In the next code cell, we demonstrate how to use this detector to find human faces in a sample image.

# In[3]:


import cv2                
import matplotlib.pyplot as plt                        
get_ipython().run_line_magic('matplotlib', 'inline')

# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

# load color (BGR) image
img = cv2.imread(human_files[1])
# convert BGR image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# find faces in image
faces = face_cascade.detectMultiScale(gray)

# print number of faces detected in the image
print('Number of faces detected:', len(faces))

# get bounding box for each detected face
for (x,y,w,h) in faces:
    # add bounding box to color image
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    
# convert BGR image to RGB for plotting
cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# display the image, along with bounding box
plt.imshow(cv_rgb)
plt.show()


# Before using any of the face detectors, it is standard procedure to convert the images to grayscale.  The `detectMultiScale` function executes the classifier stored in `face_cascade` and takes the grayscale image as a parameter.  
# 
# In the above code, `faces` is a numpy array of detected faces, where each row corresponds to a detected face.  Each detected face is a 1D array with four entries that specifies the bounding box of the detected face.  The first two entries in the array (extracted in the above code as `x` and `y`) specify the horizontal and vertical positions of the top left corner of the bounding box.  The last two entries in the array (extracted here as `w` and `h`) specify the width and height of the box.
# 
# ### Write a Human Face Detector
# 
# We can use this procedure to write a function that returns `True` if a human face is detected in an image and `False` otherwise.  This function, aptly named `face_detector`, takes a string-valued file path to an image as input and appears in the code block below.

# In[4]:


# returns "True" if face is detected in image stored at img_path
def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0


# ### (IMPLEMENTATION) Assess the Human Face Detector
# 
# __Question 1:__ Use the code cell below to test the performance of the `face_detector` function.  
# - What percentage of the first 100 images in `human_files` have a detected human face?  
# - What percentage of the first 100 images in `dog_files` have a detected human face? 
# 
# Ideally, we would like 100% of human images with a detected face and 0% of dog images with a detected face.  You will see that our algorithm falls short of this goal, but still gives acceptable performance.  We extract the file paths for the first 100 images from each of the datasets and store them in the numpy arrays `human_files_short` and `dog_files_short`.

# __Answer:__ The percentage of detected human faces is 98%
# The percentage of detected dog faces is 17%

# In[5]:


from tqdm import tqdm

human_files_short = human_files[:100]
dog_files_short = dog_files[:100]

#-#-# Do NOT modify the code above this line. #-#-#

## TODO: Test the performance of the face_detector algorithm 
## on the images in human_files_short and dog_files_short.
humans = 0
dogs = 0
for i in tqdm(human_files_short):
    if face_detector(i) == True :
        humans +=1
for i in tqdm(dog_files_short):
    if face_detector(i) == True:
        dogs+=1
print("%age of human faces in the human_short_files is:{}%".format(humans))
print("%age of human faces in the dog_files_short is:{}%".format(dogs))


# We suggest the face detector from OpenCV as a potential way to detect human images in your algorithm, but you are free to explore other approaches, especially approaches that make use of deep learning :).  Please use the code cell below to design and test your own face detection algorithm.  If you decide to pursue this _optional_ task, report performance on `human_files_short` and `dog_files_short`.

# In[6]:


### (Optional) 
### TODO: Test performance of anotherface detection algorithm.
### Feel free to use as many code cells as needed.


# ---
# <a id='step2'></a>
# ## Step 2: Detect Dogs
# 
# In this section, we use a [pre-trained model](http://pytorch.org/docs/master/torchvision/models.html) to detect dogs in images.  
# 
# ### Obtain Pre-trained VGG-16 Model
# 
# The code cell below downloads the VGG-16 model, along with weights that have been trained on [ImageNet](http://www.image-net.org/), a very large, very popular dataset used for image classification and other vision tasks.  ImageNet contains over 10 million URLs, each linking to an image containing an object from one of [1000 categories](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a).  

# In[7]:


import torch
import torchvision.models as models

# define VGG16 model
VGG16 = models.vgg16(pretrained=True)

# check if CUDA is available
use_cuda = torch.cuda.is_available()

# move model to GPU if CUDA is available
if use_cuda:
    VGG16 = VGG16.cuda()


# Given an image, this pre-trained VGG-16 model returns a prediction (derived from the 1000 possible categories in ImageNet) for the object that is contained in the image.

# ### (IMPLEMENTATION) Making Predictions with a Pre-trained Model
# 
# In the next code cell, you will write a function that accepts a path to an image (such as `'dogImages/train/001.Affenpinscher/Affenpinscher_00001.jpg'`) as input and returns the index corresponding to the ImageNet class that is predicted by the pre-trained VGG-16 model.  The output should always be an integer between 0 and 999, inclusive.
# 
# Before writing the function, make sure that you take the time to learn  how to appropriately pre-process tensors for pre-trained models in the [PyTorch documentation](http://pytorch.org/docs/stable/torchvision/models.html).

# In[8]:


from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable
def VGG16_predict(img_path):
    '''
    Use pre-trained VGG-16 model to obtain index corresponding to 
    predicted ImageNet class for image at specified path
    
    Args:
        img_path: path to an image
        
    Returns:
        Index corresponding to VGG-16 model's prediction
    '''
    
    ## TODO: Complete the function.
    ## Load and pre-process an image from the given img_path
    ## Return the *index* of the predicted class for that image
    img = Image.open(img_path)
    #input preprocessing
    transformation = transforms.Compose([ transforms.Resize((224, 224)),
                                         transforms.ToTensor(), 
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                                        ])
    img = transformation(img)
    img = img.unsqueeze(0)
    if torch.cuda.is_available():
        img = img.cuda()
    
    prediction = VGG16(img)
    if torch.cuda.is_available():
        prediction = prediction.cpu()

    return prediction.data.numpy().argmax() # predicted class index


# ### (IMPLEMENTATION) Write a Dog Detector
# 
# While looking at the [dictionary](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a), you will notice that the categories corresponding to dogs appear in an uninterrupted sequence and correspond to dictionary keys 151-268, inclusive, to include all categories from `'Chihuahua'` to `'Mexican hairless'`.  Thus, in order to check to see if an image is predicted to contain a dog by the pre-trained VGG-16 model, we need only check if the pre-trained model predicts an index between 151 and 268 (inclusive).
# 
# Use these ideas to complete the `dog_detector` function below, which returns `True` if a dog is detected in an image (and `False` if not).

# In[9]:


### returns "True" if a dog is detected in the image stored at img_path
def dog_detector(img_path):
    ## TODO: Complete the function.
    prediction = VGG16_predict(img_path) 
    if ((prediction>=151) and (prediction<=268)):
        return True
    else:
        return False


# ### (IMPLEMENTATION) Assess the Dog Detector
# 
# __Question 2:__ Use the code cell below to test the performance of your `dog_detector` function.  
# - What percentage of the images in `human_files_short` have a detected dog?  
# - What percentage of the images in `dog_files_short` have a detected dog?

# __Answer:__ 2% of images in human_files_short have a detected dog and 100% of images in dog_files_short have a detected dog

# In[10]:


### TODO: Test the performance of the dog_detector function
### on the images in human_files_short and dog_files_short.
from tqdm import tqdm
human_files_short = human_files[:100]
dog_files_short = dog_files[:100]
human_in_dog = 0
dog_in_dog = 0
for i in tqdm(human_files_short):
    if dog_detector(i) == True:
        human_in_dog +=1
for i in tqdm(dog_files_short):
    if dog_detector(i) == True:
        dog_in_dog +=1
print("%age of humans in dog files: {}%".format(human_in_dog))
print("%age of dogs in dog files: {}%".format(dog_in_dog))


# We suggest VGG-16 as a potential network to detect dog images in your algorithm, but you are free to explore other pre-trained networks (such as [Inception-v3](http://pytorch.org/docs/master/torchvision/models.html#inception-v3), [ResNet-50](http://pytorch.org/docs/master/torchvision/models.html#id3), etc).  Please use the code cell below to test other pre-trained PyTorch models.  If you decide to pursue this _optional_ task, report performance on `human_files_short` and `dog_files_short`.

# In[10]:


### (Optional) 
### TODO: Report the performance of another pre-trained network.
### Feel free to use as many code cells as needed.


# ---
# <a id='step3'></a>
# ## Step 3: Create a CNN to Classify Dog Breeds (from Scratch)
# 
# Now that we have functions for detecting humans and dogs in images, we need a way to predict breed from images.  In this step, you will create a CNN that classifies dog breeds.  You must create your CNN _from scratch_ (so, you can't use transfer learning _yet_!), and you must attain a test accuracy of at least 10%.  In Step 4 of this notebook, you will have the opportunity to use transfer learning to create a CNN that attains greatly improved accuracy.
# 
# We mention that the task of assigning breed to dogs from images is considered exceptionally challenging.  To see why, consider that *even a human* would have trouble distinguishing between a Brittany and a Welsh Springer Spaniel.  
# 
# Brittany | Welsh Springer Spaniel
# - | - 
# <img src="images/Brittany_02625.jpg" width="100"> | <img src="images/Welsh_springer_spaniel_08203.jpg" width="200">
# 
# It is not difficult to find other dog breed pairs with minimal inter-class variation (for instance, Curly-Coated Retrievers and American Water Spaniels).  
# 
# Curly-Coated Retriever | American Water Spaniel
# - | -
# <img src="images/Curly-coated_retriever_03896.jpg" width="200"> | <img src="images/American_water_spaniel_00648.jpg" width="200">
# 
# 
# Likewise, recall that labradors come in yellow, chocolate, and black.  Your vision-based algorithm will have to conquer this high intra-class variation to determine how to classify all of these different shades as the same breed.  
# 
# Yellow Labrador | Chocolate Labrador | Black Labrador
# - | -
# <img src="images/Labrador_retriever_06457.jpg" width="150"> | <img src="images/Labrador_retriever_06455.jpg" width="240"> | <img src="images/Labrador_retriever_06449.jpg" width="220">
# 
# We also mention that random chance presents an exceptionally low bar: setting aside the fact that the classes are slightly imabalanced, a random guess will provide a correct answer roughly 1 in 133 times, which corresponds to an accuracy of less than 1%.  
# 
# Remember that the practice is far ahead of the theory in deep learning.  Experiment with many different architectures, and trust your intuition.  And, of course, have fun!
# 
# ### (IMPLEMENTATION) Specify Data Loaders for the Dog Dataset
# 
# Use the code cell below to write three separate [data loaders](http://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) for the training, validation, and test datasets of dog images (located at `dog_images/train`, `dog_images/valid`, and `dog_images/test`, respectively).  You may find [this documentation on custom datasets](http://pytorch.org/docs/stable/torchvision/datasets.html) to be a useful resource.  If you are interested in augmenting your training and/or validation data, check out the wide variety of [transforms](http://pytorch.org/docs/stable/torchvision/transforms.html?highlight=transform)!

# In[11]:


import os
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# number of subprocesses to use for data loading
num_workers = 0
# how many samples per batch to load
batch_size = 20


### TODO: Write data loaders for training, validation, and test sets
## Specify appropriate transforms, and batch_sizes
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])  #normalizing data to provide data in a specific range 
#image augmentation step for training data
preprocess = transforms.Compose([
       transforms.RandomResizedCrop(224),
       transforms.RandomHorizontalFlip(),  # randomly flip and rotate
       transforms.RandomRotation(10),
       transforms.ToTensor(),              # change input image data to tensor data
       normalize
    ])
#image preprocessing for validation and test datasets
transform = transforms.Compose([
       transforms.Resize((224,224)),
       transforms.ToTensor(),normalize
    ])

train_data = datasets.ImageFolder('/data/dog_images/train', transform = preprocess)
valid_data = datasets.ImageFolder('/data/dog_images/valid', transform = transform)
test_data = datasets.ImageFolder('/data/dog_images/test', transform = transform)

train_loader = DataLoader(train_data, batch_size=batch_size,shuffle=True, num_workers=num_workers) 
valid_loader =DataLoader(valid_data, batch_size=batch_size,shuffle=True, num_workers=num_workers)
test_loader = DataLoader(test_data, batch_size=batch_size,shuffle=True, num_workers=num_workers) 

loaders_scratch = {'train': train_loader, 
                   'valid': valid_loader, 
                   'test': test_loader}


# **Question 3:** Describe your chosen procedure for preprocessing the data. 
# - How does your code resize the images (by cropping, stretching, etc)?  What size did you pick for the input tensor, and why?
# - Did you decide to augment the dataset?  If so, how (through translations, flips, rotations, etc)?  If not, why not?
# 

# **Answer**: Following are the steps that I took to preprocess the data :
# 1. Data Augmentation : I randomly crop and resize all the images to resolution of 224*224.Then to add more details to the training data i took two more data augmentation steps i.e is horizontal flipping and rotation of images. This preprocessing is done only for the training dataset.
# 2. Data preprocessing : This step is applied to all the datasets of the dog images. It involves resizing the image data to 224*224 resolution then converting it into tensor data and then normalizing the input values to a standard input range. 

# ### (IMPLEMENTATION) Model Architecture
# 
# Create a CNN to classify dog breed.  Use the template in the code cell below.

# In[12]:


import torch.nn as nn
import torch.nn.functional as F

# define the CNN architecture
class Net(nn.Module):
    ### TODO: choose an architecture, and complete the class
    def __init__(self):
        super(Net, self).__init__()
        ## Define layers of a CNN
        self.conv1 = nn.Conv2d( 3, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv4 = nn.Conv2d(64, 128, 3)
        self.conv5 = nn.Conv2d(128, 256, 3)
        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        # linear layer (256 * 5* 5 -> 500)
        self.fc1 = nn.Linear( 256 * 5 * 5, 500)
        self.fc2 = nn.Linear(500, 133)

    
    def forward(self, x):
        ## Define forward behaviorxxd
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))
        # flatten image input
        #print(x.shape)
        x = x.view(-1, 5 * 5 * 256)
        # add 1st hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        # add 2nd hidden layer, with relu activation function
        x = self.fc2(x)
        return x
import torch
#-#-# You so NOT have to modify the code below this line. #-#-#
# instantiate the CNN
model_scratch = Net()

# move tensors to GPU if CUDA is available
if use_cuda:
    model_scratch.cuda()


# __Question 4:__ Outline the steps you took to get to your final CNN architecture and your reasoning at each step.  

# __Answer:__ Based on my research i Chose to have 5 convolutional layers and 2 fully connected layers in my model.
# The first conv layers takes as input the colored image with image size of 3*224*224 and after processing gives us data with dimensions 222*222*16 as 16 is the number of filters given as input.
# 
# then this data goes through maxpool layer which down samples its size by half thus giving us the output data as 111*111*16.
# This happens at every convolutional layer and maxpool layer thus the dimensions of the input and output data for various layers is as follows :
# 
# - conv 2 : 111 * 111 * 16 --> 109 * 109 * 32
# - pool 2 : 109 * 109 * 32 --> 54 * 54 * 32
# - conv 3 : 54 * 54 * 32 --> 52 * 52 * 64
# - pool 3 : 52 * 52 * 64 --> 26 * 26 * 64
# - conv 4 : 26 * 26 * 64 --> 24 * 24 * 128
# - pool 4 : 24 * 24 * 128 --> 12 * 12 * 128
# - conv 5 : 12 * 12 * 128 --> 10 * 10 * 256
# - pool 5 : 10 * 10 * 256 --> 5 * 5 * 256
# 
# These dimensions of the convolutional layer can be calculated using the formula : (w_in - F +2P)/S+1 
# so the output of our last convolutional layer will be 5*5*256. To feed this output to the fully computational layer for classification we have to flatten this data into a vector by calling the view() function. After flattening, the data vector is fed to the fully computational network which gives us output from the 133 classes. 

# ### (IMPLEMENTATION) Specify Loss Function and Optimizer
# 
# Use the next code cell to specify a [loss function](http://pytorch.org/docs/stable/nn.html#loss-functions) and [optimizer](http://pytorch.org/docs/stable/optim.html).  Save the chosen loss function as `criterion_scratch`, and the optimizer as `optimizer_scratch` below.

# In[13]:


import torch.optim as optim

### TODO: select loss function
criterion_scratch = nn.CrossEntropyLoss()

### TODO: select optimizer
optimizer_scratch = optim.Adam(model_scratch.parameters())


# ### (IMPLEMENTATION) Train and Validate the Model
# 
# Train and validate your model in the code cell below.  [Save the final model parameters](http://pytorch.org/docs/master/notes/serialization.html) at filepath `'model_scratch.pt'`.

# In[16]:


def train(n_epochs, loaders, model, optimizer, criterion, use_cuda, save_path):
    """returns trained model"""
    # initializing minimum validation loss
    val_loss_min = np.Inf 
    
    for epoch in range(1, n_epochs+1):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        val_loss = 0.0
        
        ###################
        # train the model #
        ###################
        for batch_idx, (data, target) in enumerate(loaders['train']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            ## find the loss and update the model parameters accordingly
            ## record the average training loss, using something like
            ## train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            
            optimizer.zero_grad()  #this function is used to clear the gradients of all optimized variables
        
            output = model(data)   # responsible for predicting outputs based on the inputs provided
            
            loss = criterion(output, target)    #calculates loss 
            
            loss.backward()     # compute gradient of loss with respect to parameters
            
            optimizer.step()    #perform a single optimization step to update the parameters
            
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss)) # update the training loss
            
        ######################    
        # validate the model #
        #####################
        for batch_idx, (data, target) in enumerate(loaders['valid']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            
            output = model(data)
           
            loss = criterion(output, target)
            # update average validation loss 
            val_loss = val_loss + ((1 / (batch_idx + 1)) * (loss.data - val_loss))

    
        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, train_loss, val_loss))
        
        ## TODO: save the model if validation loss has decreased
        if val_loss <= val_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Save model ...'.format(val_loss_min, val_loss))
            torch.save(model.state_dict(), save_path)
            val_loss_min = val_loss
        
    # return trained model
    return model
# train the model
model_scratch = train(30, loaders_scratch, model_scratch, optimizer_scratch, 
                      criterion_scratch, use_cuda, 'model_scratch.pt')

# load the model that got the best validation accuracy
model_scratch.load_state_dict(torch.load('model_scratch.pt'))


# ### (IMPLEMENTATION) Test the Model
# 
# Try out your model on the test dataset of dog images.  Use the code cell below to calculate and print the test loss and accuracy.  Ensure that your test accuracy is greater than 10%.

# In[18]:


def test(loaders, model, criterion, use_cuda):

    # monitor test loss and accuracy
    test_loss = 0.
    correct = 0.
    total = 0.

    model.eval()
    for batch_idx, (data, target) in enumerate(loaders['test']):
        # move to GPU
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # update average test loss 
        test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
        # convert output probabilities to predicted class
        pred = output.data.max(1, keepdim=True)[1]
        # compare predictions to true label
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)
            
    print('Test Loss: {:.6f}\n'.format(test_loss))

    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct / total, correct, total))

# call test function    
test(loaders_scratch, model_scratch, criterion_scratch, use_cuda)


# ---
# <a id='step4'></a>
# ## Step 4: Create a CNN to Classify Dog Breeds (using Transfer Learning)
# 
# You will now use transfer learning to create a CNN that can identify dog breed from images.  Your CNN must attain at least 60% accuracy on the test set.
# 
# ### (IMPLEMENTATION) Specify Data Loaders for the Dog Dataset
# 
# Use the code cell below to write three separate [data loaders](http://pytorch.org/docs/master/data.html#torch.utils.data.DataLoader) for the training, validation, and test datasets of dog images (located at `dogImages/train`, `dogImages/valid`, and `dogImages/test`, respectively). 
# 
# If you like, **you are welcome to use the same data loaders from the previous step**, when you created a CNN from scratch.

# In[19]:


## TODO: Specify data loaders
loaders_transfer = loaders_scratch.copy() 


# ### (IMPLEMENTATION) Model Architecture
# 
# Use transfer learning to create a CNN to classify dog breed.  Use the code cell below, and save your initialized model as the variable `model_transfer`.

# In[20]:


import torchvision.models as models
import torch.nn as nn

## TODO: Specify model architecture 
model_transfer = models.resnet50(pretrained = True)

for params in model_transfer.parameters():
    params.require_grad = False

model_transfer.fc = nn.Linear(2048, 133, bias = True)

for params in model_transfer.fc.parameters():
    params.require_grad = True
    
if use_cuda:
    model_transfer = model_transfer.cuda()


# __Question 5:__ Outline the steps you took to get to your final CNN architecture and your reasoning at each step.  Describe why you think the architecture is suitable for the current problem.

# __Answer:__ I chose ResNet architecture since it is known to perform the best in cases of image classification.Also the original work that ResNet was trained on matches a lot witch this task. I chose to use all the pretrained parameters and just added one last fully computational layer to provide an output from 133 class labels provided in the dataset.
# 

# ### (IMPLEMENTATION) Specify Loss Function and Optimizer
# 
# Use the next code cell to specify a [loss function](http://pytorch.org/docs/master/nn.html#loss-functions) and [optimizer](http://pytorch.org/docs/master/optim.html).  Save the chosen loss function as `criterion_transfer`, and the optimizer as `optimizer_transfer` below.

# In[21]:


criterion_transfer = nn.CrossEntropyLoss()
optimizer_transfer = optim.Adam(model_transfer.fc.parameters())


# ### (IMPLEMENTATION) Train and Validate the Model
# 
# Train and validate your model in the code cell below.  [Save the final model parameters](http://pytorch.org/docs/master/notes/serialization.html) at filepath `'model_transfer.pt'`.

# In[22]:


# train the model
model_transfer = train(20, loaders_transfer, model_transfer, optimizer_transfer, criterion_transfer, use_cuda, 'model_transfer.pt')

# load the model that got the best validation accuracy (uncomment the line below)
model_transfer.load_state_dict(torch.load('model_transfer.pt'))


# ### (IMPLEMENTATION) Test the Model
# 
# #Try out your model on the test dataset of dog images. Use the code cell below to calculate and print the test loss and accuracy.  Ensure that your test accuracy is greater than 60%.

# In[24]:


test(loaders_transfer, model_transfer, criterion_transfer, use_cuda)


# ### (IMPLEMENTATION) Predict Dog Breed with the Model
# 
# Write a function that takes an image path as input and returns the dog breed (`Affenpinscher`, `Afghan hound`, etc) that is predicted by your model.  

# In[25]:


from PIL import Image
import torchvision.transforms as transforms

### TODO: Write a function that takes a path to an image as input
### and returns the dog breed that is predicted by the model.

data_transfer = loaders_transfer.copy()

# list of class names by index, i.e. a name can be accessed like class_names[0]
class_names = [item[4:].replace("_", " ") for item in data_transfer['train'].dataset.classes]

def predict_breed_transfer(img_path):
    global model_transfer
    # load the image and return the predicted breed

    image = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([transforms.Resize(size=(224, 224)),
                                     transforms.ToTensor()])

    # Removing transparent, alpha
    image = transform(image)[:3,:,:].unsqueeze(0)
    
    if use_cuda:
        model_transfer = model_transfer.cuda()
        image = image.cuda()
    
    model_transfer.eval()
    idx = torch.argmax(model_transfer(image))
    return class_names[idx]


# In[26]:


for img_file in os.listdir('/data/dog_images/test/001.Affenpinscher'):
    img_path = os.path.join('/data/dog_images/test/001.Affenpinscher', img_file)
    prediction = predict_breed_transfer(img_path)
    print("true label: {}, \t prediction: {}".format(img_path, prediction))


# ---
# <a id='step5'></a>
# ## Step 5: Write your Algorithm
# 
# Write an algorithm that accepts a file path to an image and first determines whether the image contains a human, dog, or neither.  Then,
# - if a __dog__ is detected in the image, return the predicted breed.
# - if a __human__ is detected in the image, return the resembling dog breed.
# - if __neither__ is detected in the image, provide output that indicates an error.
# 
# You are welcome to write your own functions for detecting humans and dogs in images, but feel free to use the `face_detector` and `human_detector` functions developed above.  You are __required__ to use your CNN from Step 4 to predict dog breed.  
# 
# Some sample output for our algorithm is provided below, but feel free to design your own user experience!
# 
# ![Sample Human Output](images/sample_human_output.png)
# 
# 
# ### (IMPLEMENTATION) Write your Algorithm

# In[27]:


### TODO: Write your algorithm.
### Feel free to use as many code cells as needed.

def run_app(img_path):
    ## handle cases for a human face, dog, and neither
    if face_detector(img_path):
        predict = predict_breed_transfer(img_path)
        print("This is the image of a human, having resemblance of the dog breed: {}".format(predict))
    elif dog_detector(img_path):
        predict = predict_breed_transfer(img_path)
        print("The breed of the dog is: {}".format(predict))
    else:
        print("Image is neithor dog, nor human")


# ---
# <a id='step6'></a>
# ## Step 6: Test Your Algorithm
# 
# In this section, you will take your new algorithm for a spin!  What kind of dog does the algorithm think that _you_ look like?  If you have a dog, does it predict your dog's breed accurately?  If you have a cat, does it mistakenly think that your cat is a dog?
# 
# ### (IMPLEMENTATION) Test Your Algorithm on Sample Images!
# 
# Test your algorithm at least six images on your computer.  Feel free to use any images you like.  Use at least two human and two dog images.  
# 
# __Question 6:__ Is the output better than you expected :) ?  Or worse :( ?  Provide at least three possible points of improvement for your algorithm.

# __Answer:__ The following things can possibly be done to improve the output of the model :
# 1. If the training loss is decreasing till the end of the numbers of epochs we can increase the number of epochs to a greater value to provide more optimization to our model.
# 2. We can also work with data augmentation process to provide more variation to the data then there already is.
# 3. We can also add more data to the datasets to provide the model with a larger dataset than provided. This will help in a better training of the model as more data generally means more learning.
# 4. We can also add more layers to our convolutional network by starting it with number of 8 filters in place of 16.This will help us extract more information in the input image.

# In[28]:


## TODO: Execute your algorithm from Step 6 on
## at least 6 images on your computer.
## Feel free to use as many code cells as needed.
import os
from PIL import Image

for img_file in os.listdir('./humans'):
    img_path = os.path.join('./humans', img_file)
    run_app(img_path)
    print(img_path)
    img = Image.open(img_path)
    plt.imshow(img)
    plt.show()
    
for img_file in os.listdir('./dogs'):
    img_path = os.path.join('./dogs', img_file)
    run_app(img_path)
    print(img_path)
    img = Image.open(img_path)
    plt.imshow(img)
    plt.show()


# In[ ]:




