Download Link: https://assignmentchef.com/product/solved-mscs18030-assignment-5-part-1-detecting-coronavirus-infections-through-chest-x-ray-images
<br>
<strong>Objectives:  </strong>

In this assignment you are required to write code for detecting infections such as COVID-19 among X-Ray images:

<ul>

 <li>Use CNN, pre-trained on ImageNet, to extract basic features from X-Ray images.</li>

 <li>Train the classification layers in order to detect instances of Infected (COVID-19 + Pneumonia) and Normal X-Ray images.</li>

 <li>Fine-tune the entire network to try to improve performance.</li>

</ul>

<strong> </strong>

This assignment must be completed using PyTorch. Assignments in <strong>Tensorflow/Keras</strong>​ or​ any other deep learning library <strong>WILL NOT BE ACCEPTED</strong>​ .​

<strong> </strong>

<h1>Chest X-Ray Images Dataset</h1>

<strong>Background: </strong>

New studies [1] have revealed that the damage done to lungs by infections belonging to the family of coronaviruses (COVID-19, SARS, ARDS etc.) can be observed in X-Ray and CT scan images. With a worldwide shortage of test kits, it is possible that careful analysis of X-Ray and CT scan images may be used in diagnosis of COVID-19 and for the regular assessment while they are recovering. In this assignment, we will use an open source dataset of X-Ray images and train a Convolutional Neural Network to try and detect instances of infections containing COVID-19 and Pneumonia.

<strong> </strong>

<strong>Dataset Details: </strong>

This dataset contains X-Ray images from 2 classes:

<table width="643">

 <tbody>

  <tr>

   <td width="157"><strong>Class </strong></td>

   <td width="171"><strong># of images in training set </strong></td>

   <td width="157"><strong># of images in validation set </strong></td>

   <td width="158"><strong># of images in test set </strong></td>

  </tr>

  <tr>

   <td width="157"><strong>Infected </strong></td>

   <td width="171">4,919</td>

   <td width="157">615</td>

   <td width="158">615</td>

  </tr>

  <tr>

   <td width="157"><strong>Normal </strong></td>

   <td width="171">7,081</td>

   <td width="157">885</td>

   <td width="158">885</td>

  </tr>

 </tbody>

</table>




Chest X-Ray images are taken in different views (AP or PA) depending on which side of the body is facing the X-Ray scanner. Images from different views have slightly different features. For this task, we will be using images without considering their views. A few sample images:

<strong> </strong>




<strong>Fine-tuning in Pytorch: </strong>

In PyTorch, each layer’s weights are stored in a Tensor. Each tensor has an attribute called ‘requires_grad’, which specifies if a layer needs training or not. In fine-tuning tasks, we freeze our pre-trained networks to a certain layer and update all the bottom layers. In PyTorch we can loop through our network layers and set ‘requires_grad’ to False for all the layers that we want to be freezed. We will set ‘requires_grad’ to True for any layer we want to fine-tune.




<strong>PyTorch Fine-tuning Tutorial: </strong><a href="https://drive.google.com/open?id=1o8va0PG6pFs3O6qJh1rRiCEgNuUG4HQm&amp;authuser=1">Lin</a><u>​ </u><a href="https://drive.google.com/open?id=1o8va0PG6pFs3O6qJh1rRiCEgNuUG4HQm&amp;authuser=1">k</a>




<strong>GitHub Repository: </strong>

Besides code and report, you are all required to make a public GitHub repository where you will upload your code and results. Following conventions are for the repository only:




<ol>

 <li>Name your code notebook as covid19_classification.ipynb</li>

 <li>Name your repository as rollNumber_COVID19_DLSpring2020</li>

 <li>Show your results in <strong>md</strong>​ (confusion matrices and accuracy)​</li>

 <li>Create a heading of Dataset and provide the link that was shared with you on Classroom.</li>

 <li>Create a folder named ‘weights’ and upload fine-tuned models. Naming convention of models is mentioned in each respective task.</li>

 <li>Add the following description to repository</li>

</ol>

“This repository contains code and results for COVID-19 classification assignment by

Deep Learning Spring 2020 course offered at Information Technology University, Lahore, Pakistan. This assignment is only for learning purposes and is not intended to be used for clinical purposes.”

<ol start="7">

 <li>You may refer to the following repository to see how they have organized their results and description:</li>

</ol>

<a href="https://github.com/kevinhkhsu/DA_detection">https://github.com/kevinhkhsu/DA_detection</a>




<h1>Task 1: Load pretrained CNN model and fine-tune FC Layers</h1>




<ul>

 <li>In this task you will fine-tune two networks <strong>(</strong>​ <strong>ResNet-18 and VGG-16)</strong> pretrained on ImageNet​</li>

 <li>Load these models in PyTorch and freeze all the layers except the last FC layers.</li>

 <li>Replace the FC layers with 2 FC layers. First FC layer will have neurons equal to:</li>

</ul>

○    (Last 2 digits of your roll number x 10) + 100

<ul>

 <li>The Last FC layer will have neurons according to the number of classes</li>

 <li>You may try different learning rates</li>

 <li>Save your model and name it as <strong>‘vgg16_FC_Only.pth’</strong>​ and ​     <strong>‘res18_FC_Only.pth’</strong>​</li>

</ul>

<strong> </strong>

<h1>Task 2: Fine-tune the CNN and FC layers of the network</h1>

<ul>

 <li>In this task you will fine-tune two pre-trained networks (ResNet-18 and VGG-16 pretrained on ImageNet weights)</li>

 <li>Perform different experiments where you first unfreeze only a few Convolutional layers and then the entire network and fine-tune on your dataset</li>

 <li>Compare the performance of training in different experiments. Show what effect it has on accuracy when you fine-tune just FC layers, then a single Conv layer, then a few Conv layers and then the entire network.</li>

 <li>Save your model where you fine-tune the whole network and name it as <strong>‘vgg16_entire.pth’</strong>​ and ‘<strong>pth’</strong>​</li>

</ul>




<strong>Requirements: </strong>

In your report, for each task, you are required to provide the following:

<ol>

 <li>Confusion Matrix for train, test and validation sets</li>

 <li>Loss and accuracy curves for train and validation sets</li>

 <li>Experimental setup (learning rate, number of layers fine-tuned etc.)</li>

 <li>Two well classified images and two worst classified images from both classes</li>

 <li>Final accuracy and F1 score for each experiment</li>

 <li>GitHub Repository link</li>

 <li>Analysis on each task and comparison of experiments to each other</li>

</ol>

<h1>Report</h1>

<ul>

 <li>Share the loss and accuracy curves on the train and validation for both.</li>

 <li>Use the same scale of axis for both the tasks so that they are comparable ● Discuss Task-1 vs Task-2, why and how it effects, which works better and why?</li>

</ul>




<strong>Please perform all tasks in the same notebook. Do NOT create separate notebooks for each task.</strong>

<strong> </strong>

<strong> </strong>

<strong>References: </strong>

<strong> </strong>

[1]  Zu, Zi Yue, et al. “Coronavirus disease 2019 (COVID-19): a perspective from China.” ​ ​<em>Radiology</em>​ (2020): 200490.


