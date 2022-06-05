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
<p>All of the code in this repository is brand new for this project, using Joseph Redmon's PyTorch tutorial (link here) as inspiration and utilizing the Kaggle Sports Images Dataset (link here). All of the implementation resides in Final_Project.ipynb (link here) from a Colab project, and contains comprehensive documentation. The notebook provides a step-by-step walk though of the project as well as an analysis of the results. This site only synthesizes the information found from the notebook.  </p>

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

<h2> :clipboard: Approach</h2>
<p>The order of execution of the program files is as follows:</p>
<p><b>1) spam_detector.py</b></p>
<p>First, the spam_detector.py file must be executed to define all the functions and variables required for classification operations.</p>
<p><b>2) train.py</b></p>
<p>Then, the train.py file must be executed, which leads to the production of the model.txt file. 
At the beginning of this file, the spam_detector has been imported so that the functions defined in it can be used.</p>
<p><b>3) test.py</b></p>
<p>Finally, the test.py file must be executed to create the result.txt and evaluation.txt files.
Just like the train.py file, at the beginning of this file, the spam_detector has been imported so that the functions defined in it can be used.</p>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<h2> :clipboard: Results</h2>

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
