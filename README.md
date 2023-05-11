# Is That Santa?

## Specific To Handin
Model files are massive so we only saved the absolute best model for reevaluation. The weights exist in the folder but it would be easier to download the dataset and rerun the experiment.

## Requirements
This project uses conda as its package manager and virtual env builder. Therefore it would be best to have conda installed on your machine to use this. However, there is also a requirements.txt file in this repo containing all dependencies. If you don't want to use conda or you don't use an m1 mac and have tensorflow already installed the way you want, you can use the requirements.txt file to create your own env with your install of python.

## Usage
The scripts that run the model, gather data, etc. are located in the top level code directory (./code). These drive the functions in further nested files. All of these scripts use arg parser to run so you can use that as a help menu to figure out commands. Likewise, with any repl, all help menus have been implemented so you can refer to those when running. 

You may have to create and activate the conda environment/ virtual environment before running. Set up conda with the command: 

```conda env create -f environment.yml```

you can activate the environment with:

```conda activate ml```

We run commands from the parent directory. From here we could train and test a model with the command:

```python code/train_main.py -g -r 1 -t```

When we are done:

```conda deactivate```

## Ways To Run/Save This Model
1. Locally
    - all data saves locally
    - make sure gcp flag is disabled when training
2. Cloud Storage
    - make sure gcp flag is enabled when training
    - you need to manually set the run number
3. As a docker container in cloud
    - update the '<SET ME>' value in the dockerfile to the run number you want to save
    - create docker image and upload it to registery of cloud storage provider you would use (recommended gcp)

## Setting Up GCP 
We used the storage buckets and vm instances to run this model. There is also a dedicated AI training platform that automates much more of the process and alerts you on status, however, it would have required a completely different setup than what we decided to use, so we chose not to use it. 

We setup one bucket that contained all output and data for the project. We kept the data in this bucket as a backup; in the event that the data did not come when created in the docker container, it could query the data itself. The bucket contains a similar folder directory structure as the one created in the local version of the project. If the folders do not exist, the model will automatically add and populate them. 

The vm instance we used was on the pricier side at $1.29 per hour, but given the time and frequency in which we used it, we only were charged $35. Our instance had the following setup: 
    - 1 NVIDIA Tesla P100 GPU 
    - 8 vCPU, 30 GB Memory
    - 50 GB Disk size
    - Debian OS with CUDA 11.3

## Related Works/ Software

- dotenv: environment variables are managed using this package. Here is a more detailed instruction on how to use it + how env variables are actually managed with it. This was used to manage the keys for the data downloading script and the google search api

- cmd2: cli interface package for python. Made it easier to run the redundant python scripts when collecting data

- cv2: bunch of different uses, but one unique one is for facial recognition command in cli. That model was found online after searching google
        --> https://github.com/chuanqi305/MobileNet-SSD/blob/master/mobilenet_iter_73000.caffemodel
- tensorflow: this is what we actually used to build the model 
- tensorflow lite: this was used to play the model on the phone for the live demo
- kaggle api: this is how we get the dataset locally (before using gcp). Requires an account but is every easy to use and free
- gcp api: this is how we interact with gcp. Requires an account and project. More complicated to set up than kaggle.

## Goal for the project
Goals:
1. in house data collection 
2. model with accuracy above 90%
3. app

## Methods
### Chosen Data Set/ Data Collection
Looking for a good premade dataset posed to be greater challenge than anticipated. There was an attempt on kaggle that compiled images of santa but we chose not to use this data set. The reason being is this dataset contains many animated images that resemble santa but not as a human being. We decided to create our own dataset. To begin this task, we first came up with a concrete description of what we considered santa. For this binary classification problem, we characterized santa as either a man with a large beard who dressed in festive clothing, or a woman who dressed in festive clothing. Race and body type did not play a factor as to whether we considered someone santa. With this in mind, we then created a few tools to facilitate the job of collecting images. Our pipeline grew to take on the following steps:

1. use a search query to pull 200 images -- limited to specific size
    - done in a repl powered by google search query api -- this is how we collected images, were bound to 100 queries a day. 
2. collected images in one place
3. ran some processing algorithms to remove noise
    --> convert to jpeg
    --> face detection for more than 1 person and 0 people
        --> many of these thresholds were just chosen on an arabitrary/ default basis
    --> renaming images
    --> removing duplicates
    --> shuffle images 
        --> provide randomness when splitting out test data
4. handpicked images that were good and bad --> had to do this twice
    -- first dataset was a little too noisey
        --> profiles, animated images, duplicates, no faces 
    -- second time looked much clearer, not as much garbage but much smaller dataset
    links to both datasets on kaggle:
        1. https://www.kaggle.com/datasets/ianmaloney/santa-images
        2. https://www.kaggle.com/datasets/ianmaloney/updated-santa

Dataset included some of the following searches:
 - people who look like santa
 - man in santa suit
 - santa 
 - white bearded man
 - santa claus inpersonator
 - santa lookalike
 - santa human
 - santa costume
 - mall santa
 - gandalf
 - black santa
 - russian santa
 - black man with white beard
 - hispanic santa 
 - european santa 
 - african santa 
 - asian santa
 - santa woman

 To build our negative set, we used the following searches:
 - person face
 - random person 
 - person portrait 
 - white man
 - black man 
 - white woman
 - black woman
 - asian man 
 - asian woman
 - african man
 - african woman
 - hispanic man
 - hispanic woman

 As expected, we received higher quality images collecting the negative set rather than the positive set

Success for a search is defined by the quality of images returned. The search queries above gave the highest quality images in terms of what was being captured, but they do not represent an exhaustive list. Many searches led to incorrect results. Ironically, efforts to make the model more inclusionary resulted in the inclusion of racist images, caricatures, and controversial historical characters. These added quite a bit of noise. We elaborate more on this in the model section, where we discuss techniques used to increase accuracy. Ultimately, some stricter contraints on who could be considered santa were eventually invoked. 

Some metrics about each dataset:
    dataset 1: 1420 total images (training: 610 santa, 610 not santa | testing: 100 santa, 100 not santa)
        --> this dataset is is more robust with a larger image count, but some of its images are noisey, 
        leading to some poor overfitting when testing this
    dataset 2: 793 total images (training: 355 not santa, 300 santa | tetsting: 65 not santa, 73 santa)
        --> smaller dataset so it was a concern that the data may be overfit, but the data was much more consitent with what was considered appropriate for a santa claus example

Aside from image quality, one of the key differences is the imbalance in the second dataset compared to the balanced nature of the first. We talk about this more in the later section, but briefly, this is overcome by weighting the data we train with.


### Model
Given the nature of the problem we are trying to solve. Our goal for an acceptable accuracy was 90% or higher. We chose 90% because we are answering a binary classification problem where a random guess places 50% likelihood of either label being chosen. This is a subjective kind of problem so we want to give our model some leeway in its accuracy, hence the choice of 90%. In terms of the model, we used various techniques to both improve the performance, and capture valuable metrics for model evaluation. We discuss both in brief following:

#### Techniques:
The following is a list of techniques we used to improve accuracy, along with a minor description and how it was used:
    - learning rate scheduler -- we opted to use an increasing learning rate proportionate to the number of epochs used
    - Early Stopping -- If the loss remained stagnant or increased over a number of epochs, training would end. This saved us time in that if a model were becoming grossly overfit, we would just stop training immediately. We call this value our patience value. We began with a patience value of 8, but decreased it to 3 overtime.
    - data augmentation -- as mentioned, how we derived some of the values for data augmentation was lost due to poor logging, yet the final set was as follows:

        brightness_range = [0.1, 0.9],
        channel_shift_range = 0.85,
        horizontal_flip = True,
        rotation_range = 45,
        vertical_flip = True,
        shear_range = 0.8,
        zoom_range = 0.5,
        height_shift_range = 35,
        width_shift_range = 35,
        rescale = 1.0/255.0,
        samplewise_center = True,
        validation_split = .18,
        preprocessing_function = self.preprocess_data,

    Many of these values were chosen arbitrarily and kept if they affected the difference between the testing accuracy and validation accuracy on the best run. We will discus samplewise_center and self.preprocess_data further. 

    #### self.preprocess_data
    TODO: try it as greyscale
    To remain consistent across all images, we convert each sample to an rgb image. We assumed santa's festive clothing may influence the model so we decided to keep color. In the event a sample was either greyscale or rgba, we would convert it to rgb.

    #### samplewise center
    each individual image is independently normalized. Given that our images are pretty different both due to their own general appearances and the data augmentation process, we felt it would make sense to normalize them according to their own information rather than a mean of images (which could be sensitive to outliers).

### Metrics
Given the fact that we are solving a binary classification problem, we used the following metrics to determine progress for our model. Loss has its own dedicated section. Adjacent is a small description of the metrics, how it is calculated, and what we were looking for:

Accuracy - we track accuracy, and it is important in saving the model checkpoints, however, one thing we were concerned about is the bias introduced with an imbalanced dataset. Since we have more negative examples, the model could overstate its accuracy in the training and validation set if it always picks not santa. In order to combat this, we weighted the positive and negative examples by their presence in the dataset. Even still we focus on other metrics for a better picture.
Learning Rate - we use a logistic smoothing function to increase the learning rate from a min learning rate to a max learning rate value proportional to the amount of epochs. We used the formula: 
min_lr + 0.99 * (max_lr - min_lr) * (1 / (1 + np.exp(-0.1 * (epoch - 10))))
True Positives - number of correct positive identifications. We want to maximize this
False Positives - number of incorrect positive identifications. We want to minimize this
True Negatives - number of correct negative identifications. We want to maximize this
False Negatives - number of incorrect negative identifications. We want to minimize this
Precision - precise the model is when predicting the positive class
    TP / (TP + FP)
F1 Score - balanced measure of the model's performance in terms of both precision and recall.
    2 * (Precision * Recall) / (Precision + Recall)
Recall - model's ability to identify all positive instances
    TP / (TP + FN)

### Hyperparameters
The model using the following hyperparamerters to run the model
min_learning_rate = 1e-05
max_learning_rate = 0.001
num_epochs = 50
batch_size = 15
image_size = 256
num_classes = 2
validation_split = 0.18
leaky_relu_alpha = 0.3
dropout = 0.7
gamma = 3.0
momentum = 0.8
patience = 5
test_batch_size = 6
l2 = 0.01


### Architecture
In conjunction with all prior details listed, we implemented two models to see which would perform better.
1. transfer model backed by InceptionV3 
    --> choice was between Xception and MobileNetV2
    --> focused on Inception
        --> we already used mobile net for face recognition before, dealt with smaller images
        --> Xception designed to be faster but not necessarily more performant (and we aren't training the model)
    
2. simpler model made up of many different convolutional layers
    --> messed with number of layers, amount, etc.
    Conv2D(3, 3,activation=tf.keras.layers.LeakyReLU(alpha=hp.leaky_relu_alpha), input_shape=(hp.image_size, hp.image_size, 3))
    1. conv layer
        --> kernel size: (3,3)
        --> filters: 3
        --> activation: Leaky Relu
    2. conv layer
        --> kernel size: (4,4)
        --> filters: 3
        --> activation: Leaky Relu 
    3. conv layer
        --> kernel size: (5, 5)
        --> filters: 30
        --> activation: Leaky Relu
    4. conv layer
        --> kernel size: (3, 3)
        --> filters: 100
        --> activation: Leaky Relu
    5. conv layer
        --> kernel size: (4, 4)
        --> filters: 250
        --> activation: Leaky Relu
    6. Dense Layer
        --> neurons: 10
        --> activation: Leaky Relu
    6. Dense layer
        --> neurons: 1
        --> activation: sigmoid


#### Activation Functions

Leaky Relu:

Sigmoid:

#### Loss function
We used the same loss across both models. The chosen loss was BinaryFocalCrossEntropy instead of BinaryCrossEntropy. BinaryFocalCrossEntropy [adds a focal factor that down-weights easy examples and focuses more on hard examples](https://www.tensorflow.org/api_docs/python/tf/keras/losses/BinaryFocalCrossentropy). We felt it was most appropriate to give weight to harder examples because we were using an unbalanced dataset. Therefore, it would be harder to tell the minority group from the majority since there are many more examples of the majority.

## Logging
Great efforts were made to record as much data per run as possible. We have different categories for logs written out by the folder they are saved to. Below is a small description:
1. data_augmentation -- parameters used for data augmentation, added more as time went on
2. hyperparameters -- capture of the chosen hyper parameters
3. info -- information about each epoch
4. learning_rate -- how the learning rate changes over time
5. logs -- things for tensorboard
6. model_logs -- screencapture of the model for rerunning
7. saved_models -- the saved model (uses this to convert for app)
8. summaries -- parameter lists for model
9. test_results -- results from test data
10. transfer_model_logs -- screencapture of transfer mode
google search query -- limited to 200 images sized to a specific amount
had to try a bunch of different search queries 
