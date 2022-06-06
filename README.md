<p align="center">
  <a href="https://github.com/garrettdevereux/ComputerVisionFinal"><img src="https://img.shields.io/badge/Repo----blue"></a>
  <a href="https://github.com/garrettdevereux/ComputerVisionFinal/blob/main/Final_Project.ipynb"><img src="https://img.shields.io/badge/Code----green"></a>
  <a href="https://www.kaggle.com/datasets/gpiosenka/sports-classification?resource=download"><img src="https://img.shields.io/badge/Dataset----yellow"></a>
  <a href="https://github.com/garrettdevereux/ComputerVisionFinal"><img src="https://img.shields.io/badge/Video----red"></a>
</p>

<p align="center"> 
  <center><img src="website/netImage.png" alt="Net Image.png" width="80px" height="80px"></center>
</p>
<h1 align="center"> Sports Images Classifier </h1>
<h3 align="center"> CSE 455 - Computer Vision </h3>
<h5 align="center"> Final Project - Garrett Devereux - <a href="https://courses.cs.washington.edu/courses/cse455/22sp/">University of Washington</a> (Spring 2022) </h5>

<h2> <img src="website/problemDescription.png" alt="Problem Description.png" width="50px" height="50px"> Problem Description</h2>
<p>Sports have always been an important part of my life, providing me with the motivation to constantly improve my skills and health, help build releationships and connections with others, and teach me lessons about teamwork and perserverance that carry over into my other pursuits. My project directly relates to this passion and aims to analyze a variety of images containing sports and classify them using neural networks and computer vision. My data set comes from this <a href="https://www.kaggle.com/datasets/gpiosenka/sports-classification?resource=download">Kaggle Competition</a> which contains over 14,000 images from 100 different sports. In my project, I explore a variety of different model architectures as well as different data processesing and training techniques such as data augmentation and transfer learning. I then compare and contrast each models performance and see how they do in the real world with pictures of myself playing sports throughout the years. </p>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<h2><img src="website/preexisting.png" alt="Preexisting Work.png" width="50px" height="50px"> Preexisting Work</h2>
<p>All of the code in this repository is brand new for this project, using Professor Joseph Redmon's PyTorch tutorial (link here) as inspiration and utilizing the Kaggle Sports Images Dataset (link here). All of the implementation resides in Final_Project.ipynb (link here) from a Colab project, and contains comprehensive documentation. The notebook provides a step-by-step walk though of the project as well as an analysis of the results. This site only synthesizes the information found from the notebook.  </p>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<h2> <img src="website/projectStructure.webp" alt="Project Structure.png" width="50px" height="50px"> Project Files</h2>

All files from this project can be found in the <a href="https://github.com/garrettdevereux/ComputerVisionFinal">Repository</a>.

<p>This Project includes a Colab project with all of the code, a dataset folder, and a results folder.</p>
<h4>Code:</h4>
<ul>
  <li><b>Final_Project.ipynb</b> - Includes all of the implementation for the project included with a detailed walkthrough and analysis.</li>
</ul>

<h4>Dataset: in the <b>sportsImages/</b> folder</h4>
<ul>
  <li><b>sportsImages/models/</b> - Contains all of the trained models produced throughout the project, as well as .csv files showing each models prediction versus the actual label on the entire test set. *Note: the simple model was too large of a file to handle, so it is not included.</li>
  <li><b>sportsImages/realworld/</b> - Contains twenty different images of myself playing sports over the years. Used to check the models performance in the real world.</li>
  <li><b>sportsImages/test/</b> - Contains all 500 images of the test set. Inside /test/ are 100 folders labeled with the corresponding sport, and then inside each sport folder is five images.</li>
  <li><b>sportsImages/train/</b> - Contains all 13572 images of the training set. Inside /train/ are 100 folders labeled with the corresponding sport, and then inside each sport folder is on average 136 images.</li>
  <li><b>sportsImages/valid/</b> - Contains all 500 images of the validation set. Inside /valid/ are 100 folders labeled with the corresponding sport, and then inside each sport folder is five images.</li>
  <li><b>sportsImages/class_dict.csv</b> - A .csv file containing an entry for each sport in the form: index, sport name, height, width, scale. This allows us to take a prediction index and map it to a class name.</li>
</ul>

<h4><b>Results/</b>:</h4>
<ul>
  <li><b>Results/ComparisonResults/</b> - Includes the 10 images randomly selected from the validation batch, as well as each of the five models predictions for each image in <b>validResults.txt</b>. Also contains the file <b>realworldResults.txt</b> which contains each of the five models predictions for the twenty images in the realworld dataset.</li>
  <li><b>Results/DatasetExampleImages/</b> - Includes two examples of a batch of 32 images from the training set with different augmentations applied.</li>
  <li><b>Results/TrainingResults/</b> Contains a graph of the training loss for each of the five models over 20 epochs, as well as a .txt file with the exact loss after every 10 batches during training. </li>
</ul>

<h4>Other:</h4>
<ul>
  <li><b>website/</b> - Includes all assets for the website.</li>
  <li><b>README.md</b> - This document. Includes the code for the website.</li>
  <li><b>_.config.yml</b> - Config file for the webpage.</li>
</ul>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<h2> <img src="website/projectStructure.webp" alt="Project Structure.png" width="50px" height="50px">Dataset and Preprocessing</h2>

<p>The sportsImages dataset contains images from 100 different sports from air hockey to wingsuit flying. Reference the class_dict.csv (link here) for all of the classes. There are 13572 images in the training set, and 500 in both the test and validation set. The images are 3x224x224 which I then size down to 3x150x150 to help reduce the dimensions and train faster. This greatly reduces the size of the simple model, which is still too big to store on github. On my first attempt of training the models, I performed a resize, random crop, random horizontal flip, random rotation, and then added gaussian noise to each image. 32 of these images looked like:</p>
<p align="center"> 
  <center><img src="Results/DatasetExampleImages/DatasetFirstTry.png" alt="DatasetFirst.png"></center>
</p>
<p>After training the models for the first time, I found the test accuracy was terrible, resulting in around 10% test accuracy for the simple model, 50% for the convolutional model, and 37% for the pretrained resnet18. It was clear that this augmentation was too different from the test set and was hurting the performance much more than it was helping. So, after experimenting with a variety of other augmentations, I finally found the right mix of Gaussian noise and rotation range to get the following: </p>
<p align="center"> 
  <center><img src="Results/DatasetExampleImages/Dataset.png" alt="Dataset.png"></center>
</p>
<p>In order to load this dataset and perform the augmentations, view the sections 'Loading the Dataset', 'Data Augmentation', and 'Understanding the Dataset' in Final_Project.ipynb. These sections will also provide more in depth explanations of how to access and manipulate images, as well as how to print out batches like the two above.</p>


![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<h2> <img src="website/projectStructure.webp" alt="Project Structure.png" width="50px" height="50px">Approach</h2>
<p><b> Building networks for classification </b></p>

<p><b> SimpleNet </b></p>
<p>I first approached the problem by building a very simple model to get a baseline accuracy score and an idea of how comlplex the dataset is to learn. This simple model is called SimpleNet, and is a very vanilla Neural Network with a single hidden layer using leak relu activation. As the input images are augmented down to (3 x 150 x 150) the input size will be 67500. Additionally, since the model is making predictions for 100 classes, the final layer needs to be of size 100. With a hidden layer size of 512, there will be 34,611,200 (67500x512 + 512x100) different weights in the model. </p>

<p><b> Convnet </b></p>
<p>As a better approach, I next worked to take advantage of the structure of images with convolutions, batch normalization, and pooling to greatly reduce the size of the network while increasing its power. Here I referenced Professor Redmon's demo lecture on Convolutional Neural Networks and chose to use the Darknet Architecture (link). As my images we much larger than the example (3x150x150 vs 3x64x64), I worked to tweak the strides of the different convolutions to produce a reasonably size result. This resulted in five convolutions taking the (3x150x150)->(16x50x50)->(32x25x25)->(64x13x13)->(128x7x7)->(256x4x4). Additionally, I changed the final output layer to size 100 to fits the number of classes in my dataset. </p>

<p><b> Resnetv1 </b></p>
<p> To continue to increase model performance, I then moved to using transfer learning on pretrained networks. As a first try of this, I again followed Redmon's demo and fit the pretrained resnet18 model to my own dataset. This required me to change the final fully connected layer to map to 100 classes rather than the over 20,000 categories of ImageNet.</p>

<p><b> Resnetv2 and Effnet </b></p>
<p>Finally, I wanted build a model better than all of the above by taking advantage of the techniques that worked and continuing to tweak the parameters and process. As transfer learning worked the best, I first worked to experiment with the data augmentation to see if I could find transformations that would produce better results. Using the exact same resnet18 architecture referenced above, I tried several different augmentations, adding more and less rotation and noise, removing the flipping and cropping of images, and in the end I found that keeping the horizontal image flips, removing the rotations and noise, and normalizing the image before putting it through the network worked the best. With this knowledge, I then again took advantage of transfer learning to bring in the Efficientnet_b0 with the new set of augmentations.</p>

<p> For futher reference, each of these models is implemented in Final Project (link) with full explanation and results.

<p><b>Models Implemented:</b></p>
<ul>
						<li>SimpleNet</li>
							<ul>
								<li>1 Hidden Layer with Leaky Relu Activation</li>
								<li>Structure:</li>
									<ul>
										<li>Linear(67500, 512)</li>
										<li>Linear(512, 100)</li>
									</ul>
							</ul>
						<li>ConvNet</li>
							<ul>
								<li>DarkNet Architecture: 5 Convolutional Layers with batch normalization and a linear layer</li>
								<li>Structure:</li>
									<ul>
										<li>Conv2d(3, 16, 3, stride=3, padding=1)</li>
										<li>BatchNorm2d(16)</li>
										<li>Conv2d(16, 32, 3, stride=2, padding=1)</li>
                    <li>BatchNorm2d(32)</li>
										<li>Conv2d(32, 64, 3, stride=1, padding=1)</li>
										<li>BatchNorm2d(64)</li>
										<li>Conv2d(64, 128, 3, stride=2, padding=1)</li>
										<li>BatchNorm2d(128)</li>
										<li>Conv2d(128, 256, 3, stride=2, padding=1)</li>
										<li>BatchNorm2d(256)</li>
										<li>Linear(256, 100)</li>
									</ul>
							</ul>
						<li>Resnetv1 and Resnetv2</li>
							<ul>
                <li>Structure:</li>
							</ul>
              <li>Efficientnet_b0</li>
							<ul>
								<li>Structure:</li>
							</ul>
					</ul>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<h2> :clipboard: Results</h2>

<p>After training the models several times with different parameters, I ended with the following results: </p>

<p><b> SimpleNet: </b></p>
<p> 20 Epochs, LR shedule: {0:0.01, 15: 0.001}, Batch Size: 128.</p>
<p> <img src="Results/TrainingResults/simpleTraining.png" alt="simpleTraining.png"></p>
<p> Performance: Training loss ending at 3.4. <b>20.2% Testing Accuracy</b>. See (here) for full loss data.

<p><b> ConvNet: </b></p>
<p> 20 Epochs, LR shedule: {0:0.1, 5:0.01, 15: 0.001}, Batch Size: 128.</p>
<p> <img src="Results/TrainingResults/convnetTraining.png" alt="convnetTraining.png"></p>
<p> Performance: Training loss ending at 1.2. 66.4% Testing Accuracy after 20 epochs. <b>67.0% accuracy after 17 epochs</b>. See (here) for full loss data.

<p><b> Resnetv1: </b></p>
<p> 20 Epochs, LR shedule: {0:0.1, 5:0.01, 15: 0.001}, Batch Size: 128.</p>
<p> <img src="Results/TrainingResults/resnetv1Training.png" alt="resnetv1Training.png"></p>
<p> Performance: Training loss ending at 0.1. <b>92.2% Testing Accuracy after 20 epochs</b>. See (here) for full loss data.

<p><b> Resnetv2: </b></p>
<p> 20 Epochs, LR shedule: {0:0.1, 5:0.01, 15: 0.001}, Batch Size: 128.</p>
<p> <img src="Results/TrainingResults/resnetv2Training.png" alt="resnetv2Training.png"></p>
<p> Performance: Training loss ending at 0.015. <b>94.6% Testing Accuracy after 20 epochs</b>. See (here) for full loss data.

<p><b> EffNet: </b></p>
<p> 20 Epochs, LR shedule: {0:0.1, 5:0.01, 15: 0.001}, Batch Size: 128.</p>
<p> <img src="Results/TrainingResults/effnetTraining.png" alt="effnetTraining.png"></p>
<p> Performance: Training loss ending at 0.018. 95.6% Testing Accuracy after 20 epochs. <b>95.8% accuracy after 17 epochs</b>. See (here) for full loss data.

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<h2> :clipboard: Comparison</h2>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<h2> :clipboard: Discussion</h2>
<p>Got kicked off GPU quick and spent lots of time waiting for training. 
Really cool to figure out how to do everything on my own, I feel that
I could confidently complete my own independent project.</p>


![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<h2> <img src="website/references.png" alt="Problem Description.png" width="50px" height="50px">  References</h2>
<ul>
  <li><p>Joseph Redmon, 'iPynb Tutorial Series'. [Online].</p>
      <p>Available: https://courses.cs.washington.edu/courses/cse455/22sp/</p>
  </li>
  <li><p>Gerry's Kaggle Dataset, '100 Sports Image Classification'. [Online].</p>
      <p>Available: https://www.kaggle.com/datasets/gpiosenka/sports-classification?resource=download</p>
  </li>
  <li><p>Mohammad Amin Shamshiri, 'READMD.md template'. [Online].</p>
      <p>Available: https://github.com/ma-shamshiri/Spam-Detector/blob/master/README.md</p>
  </li>
</ul>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<!-- CREDITS -->
<h2 id="credits"> :scroll: Credits</h2>

Garrett Devereux

[![GitHub Badge](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/garrettdevereux)
[![LinkedIn Badge](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/garrett-devereux/)
