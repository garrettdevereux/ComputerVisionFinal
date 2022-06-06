<p align="center">
  <a href="https://github.com/garrettdevereux/ComputerVisionFinal"><img src="https://img.shields.io/badge/Repo----blue"></a>
  <a href="https://github.com/garrettdevereux/ComputerVisionFinal/blob/main/Final_Project.ipynb"><img src="https://img.shields.io/badge/Code----green"></a>
  <a href="https://www.kaggle.com/datasets/gpiosenka/sports-classification?resource=download"><img src="https://img.shields.io/badge/Dataset----yellow"></a>
  <a href="https://github.com/garrettdevereux/ComputerVisionFinal"><img src="https://img.shields.io/badge/Video----red"></a>
</p>

<p align="center"> <img src="website/netImage.png" alt="Net Image.png" width="80px" height="80px"></p>
<h1 align="center"> Sports Images Classifier </h1>
<h3 align="center"> CSE 455 - Computer Vision </h3>
<h5 align="center"> Final Project - Garrett Devereux - <a href="https://courses.cs.washington.edu/courses/cse455/22sp/">University of Washington</a> (Spring 2022) </h5>

<h2> <img src="website/problemDescription.png" alt="Problem Description.png" width="50px" height="50px"> Problem Description</h2>
<p>Sports have always been an important part of my life, providing me with the motivation to constantly improve my skills and health, help build relationships and connections with others, and teach me lessons about teamwork and perseverance that carry over into my other pursuits. My project directly relates to this passion and aims to analyze a variety of images containing sports and classify them using neural networks and computer vision techniques. My data set comes from this <a href="https://www.kaggle.com/datasets/gpiosenka/sports-classification?resource=download">Kaggle Competition</a> which contains over 14,000 images from 100 different sports. In my project, I explore a variety of different model architectures and data processing and training techniques such as data augmentation and transfer learning. I then compare and contrast each model’s performance and see how they do in the real world with pictures of myself playing sports throughout the years.</p>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<h2><img src="website/preexisting.png" alt="Preexisting Work.png" width="50px" height="50px"> Preexisting Work</h2>
<p>All of the code in this repository is brand new for this project, using Professor Joseph Redmon's PyTorch tutorial as inspiration and utilizing the Kaggle Sports Images Dataset. The entirety of the implementation resides in <a href="https://github.com/garrettdevereux/ComputerVisionFinal/blob/main/Final_Project.ipynb">Final_Project.ipynb</a> as a Colab notebook, and contains comprehensive documentation. The notebook provides a step-by-step walk though of the project as well as an analysis of the results. This site only synthesizes the information found in the notebook.</p>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<h2> <img src="website/projectStructure.webp" alt="Project Structure.png" width="50px" height="50px"> Project Files</h2>

All files from this project can be found in the <a href="https://github.com/garrettdevereux/ComputerVisionFinal">Repository</a>. The video presentation is linked at the top of this webpage and can be found <a href="https://github.com/garrettdevereux/ComputerVisionFinal">here</a>.

<p>This Project includes a Colab notebook with all of the code, a dataset folder, and a results folder.</p>
<h4>Code:</h4>
<ul>
  <li><b>Final_Project.ipynb</b> - Includes all of the implementation for the project with a detailed walkthrough and analysis.</li>
</ul>

<h4>Dataset: in the <b>sportsImages/</b> folder</h4>
<ul>
  <li><b>sportsImages/models/</b> - Contains all of the trained models produced throughout the project, as well as .csv files showing each model's prediction versus the actual label on the entire test set. *Note: the simple model was too large of a file to handle, so it is not included.</li>
  <li><b>sportsImages/realworld/</b> - Contains twenty different images of myself playing sports over the years. Used to check each model's performance in the real world.</li>
  <li><b>sportsImages/test/</b> - Contains all 500 images of the test set. Inside /test/ are 100 folders labeled with the corresponding sport, and then inside each sport folder is five images.</li>
  <li><b>sportsImages/train/</b> - Contains all 13572 images of the training set. Inside /train/ are 100 folders labeled with the corresponding sport, and then inside each sport folder is on average 136 images.</li>
  <li><b>sportsImages/valid/</b> - Contains all 500 images of the validation set. Inside /valid/ are 100 folders labeled with the corresponding sport, and then inside each sport folder is five images.</li>
  <li><b>sportsImages/class_dict.csv</b> - A .csv file containing an entry for each sport in the form: index, sport name, height, width, scale. This allows us to take a prediction index and map it to a class name.</li>
</ul>

<h4><b>Results/</b>:</h4>
<ul>
  <li><b>Results/ComparisonResults/</b> - Includes 11 images for the validation comparison with each model's prediction, and twenty images for the real world comparison with each of the five model's predictions.</li>
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

<h2> <img src="website/preprocess.jpg" alt="process.png" width="50px" height="50px">Dataset and Preprocessing</h2>

<p>The sportsImages dataset contains images from 100 different sports from air hockey to wingsuit flying. Reference the <a href="https://github.com/garrettdevereux/ComputerVisionFinal/blob/main/sportsImages/class_dict.csv">class_dict.csv</a> for all of the classes. There are 13572 images in the training set, and 500 in both the test and validation set. The images are 3x224x224 which I then size down to 3x150x150 to help reduce the dimensions and train faster. This greatly reduces the size of the simple model, which is still too big to store on github. On my first attempt of training the models, I performed a resize, random crop, random horizontal flip, random rotation, and then added gaussian noise to each image. 32 of these images looked like:</p>
<p align="center"> 
  <center><img src="Results/DatasetExampleImages/DatasetFirstTry.png" alt="DatasetFirst.png"></center>
</p>
<p>After training the models for the first time with the above augmentation, I found the test accuracy was terrible, with the simple model acheiving around 10% test accuracy, 50% for the convolutional model, and only 37% for the pretrained resnet18. It was clear that this augmentation was doing too much alteration from the test set and was hurting the performance much more than it was helping. So, after experimenting with a variety of other augmentations, I finally found the right mix of Gaussian noise and rotation range to get the following: </p>
<p align="center"> 
  <center><img src="Results/DatasetExampleImages/Dataset.png" alt="Dataset.png"></center>
</p>
<p>In order to load this dataset and perform the augmentations, view the sections 'Loading the Dataset', 'Data Augmentation', and 'Understanding the Dataset' in <a href="https://github.com/garrettdevereux/ComputerVisionFinal/blob/main/Final_Project.ipynb">Final_Project.ipynb</a>. These sections will also provide more in depth explanations of how to access and manipulate images, as well as how to print out batches like the two above.</p>


![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<h2> <img src="website/approach.png" alt="app.png" width="50px" height="50px">Approach</h2>
<p><b> Building networks for classification </b></p>

<p><b> SimpleNet </b></p>
<p>I first approached the problem by building a very simple model to get a baseline accuracy score and an idea of how complex the dataset is to learn. This simple model is called SimpleNet and is a very vanilla Neural Network with a single hidden layer using Leaky-Relu activation. As the input images are augmented down to (3 x 150 x 150) the input size will be 67500. Additionally, since the model is making predictions for 100 classes, the final layer needs to be of size 100. With a hidden layer size of 512, there will be 34,611,200 (67500x512 + 512x100) different weights in the model. </p>

<p><b> ConvNet </b></p>
<p>As a better approach, I next worked to take advantage of the structure of images with convolutions, batch normalization, and pooling to greatly reduce the size of the network while increasing its power. Here I referenced Professor Redmon's demo lecture on Convolutional Neural Networks and chose to use the <a href="https://pjreddie.com/darknet/imagenet/#reference">Darknet Architecture</a>. As my images were much larger than the example (3x150x150 vs 3x64x64), I worked to tweak the strides and sizes of the different convolutions to produce a reasonably sized outcome. This resulted in five convolutions mapping the input image to the following sizes: (3x150x150)->(16x50x50)->(32x25x25)->(64x13x13)->(128x7x7)->(256x4x4). Additionally, I changed the final output layer to size 100 to fit the number of classes in my dataset. </p>

<p><b> ResNetv1 </b></p>
<p> To continue to increase model performance, I then moved to using transfer learning on pretrained networks. As a first try of this, I again followed Redmon's demo and fit the pretrained resnet18 model to my own dataset. This required me to change the final fully connected layer to map to 100 classes rather than the over 20,000 categories of ImageNet.</p>

<p><b> ResNetv2 and EffNet </b></p>
<p>Finally, I wanted to build a model better than all of the above by taking advantage of the techniques that worked and continuing to tweak the parameters and process. As the Resnet transfer learning worked the best, I first wanted to use the same model with different data augmentation to see if I could find transformations that would produce better results. Using the exact same resnet18 architecture referenced above, I tried several new augmentations, adding more and less rotation and noise, removing the flipping and cropping of images, and in the end, I found that keeping the horizontal image flips, removing the rotations and noise, and normalizing the image before putting it through the network worked the best. With this knowledge, I then again took advantage of transfer learning to bring in the Efficientnet_b0 with the new set of augmentations.</p>

<p><b>Models Implemented:</b></p>
<ul>
	<li>SimpleNet</li>
	<ul>
		<li>1 Hidden Layer with Leaky-Relu Activation</li>
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
			<li>Conv2d(32, 64, 3, stride=2, padding=1)</li>
			<li>BatchNorm2d(64)</li>
			<li>Conv2d(64, 128, 3, stride=2, padding=1)</li>
			<li>BatchNorm2d(128)</li>
			<li>Conv2d(128, 256, 3, stride=2, padding=1)</li>
			<li>BatchNorm2d(256)</li>
			<li>Linear(256, 100)</li>
		</ul>
	</ul>
	<li>ResNetv1 and ResNetv2</li>
	<ul>
    <li>Structure:</li>
		<img src="website/resnet18_arch.png" alt="res18.png" width="500px">
		<img src="website/resnet18-diag.png" alt="res18.png" width="1000px">
	</ul>
  <li>EffNet</li>
	<ul>
		<li>Structure:</li>
		<img src="website/effnetb0.png" alt="eff.png" width="600px">
		<img src="website/effnetb0_arch.png" alt="eff.png" width="800px">
	</ul>
</ul>

<p> For futher reference, each of these models is implemented in <a href="https://github.com/garrettdevereux/ComputerVisionFinal/blob/main/Final_Project.ipynb">Final_Project.ipynb</a> with full explanation and results.</p>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<h2> <img src="website/results.png" alt="res.png" width="50px" height="50px"> Results</h2>

<p>After training the models several times with different parameters, I ended with the following results: </p>

<p><b> SimpleNet: </b></p>
<p> 20 Epochs, Learning Rate schedule: {0:0.01, 15: 0.001}, Batch Size: 128.</p>
<p> <img src="Results/TrainingResults/simpleTraining.png" alt="simpleTraining.png"></p>
<p> Performance: <a href="https://github.com/garrettdevereux/ComputerVisionFinal/blob/main/Results/TrainingResults/simpleTraining.txt">Training Loss</a> ended at 3.4. <b>20.2% Testing Accuracy</b>. <a href="https://github.com/garrettdevereux/ComputerVisionFinal/blob/main/sportsImages/models/simple20Preds.csv"> Predictions versus Actual labels</a>.</p>

<p><b> ConvNet: </b></p>
<p> 20 Epochs, Learning Rate schedule: {0:0.1, 5:0.01, 15: 0.001}, Batch Size: 128.</p>
<p> <img src="Results/TrainingResults/convnetTraining.png" alt="convnetTraining.png"></p>
<p> Performance: <a href="https://github.com/garrettdevereux/ComputerVisionFinal/blob/main/Results/TrainingResults/convnetTraining.txt">Training Loss</a> ended at 1.2. 66.4% Testing Accuracy after 20 epochs. <b>67.0% accuracy after 17 epochs</b>. <a href="https://github.com/garrettdevereux/ComputerVisionFinal/blob/main/sportsImages/models/convnet17Preds.csv"> Predictions versus Actual labels</a>.</p>

<p><b> ResNetv1: </b></p>
<p> 20 Epochs, Learning Rate schedule: {0:0.1, 5:0.01, 15: 0.001}, Batch Size: 128.</p>
<p> <img src="Results/TrainingResults/resnetv1Training.png" alt="resnetv1Training.png"></p>
<p> Performance: <a href="https://github.com/garrettdevereux/ComputerVisionFinal/blob/main/Results/TrainingResults/resnetv1Training.txt">Training Loss</a> ended at 0.1. <b>92.2% Testing Accuracy after 20 epochs</b>. <a href="https://github.com/garrettdevereux/ComputerVisionFinal/blob/main/sportsImages/models/resnet20Preds.csv"> Predictions versus Actual labels</a>.</p>

<p><b> ResNetv2: </b></p>
<p> 20 Epochs, Learning Rate schedule: {0:0.1, 5:0.01, 15: 0.001}, Batch Size: 128.</p>
<p> <img src="Results/TrainingResults/resnetv2Training.png" alt="resnetv2Training.png"></p>
<p> Performance: <a href="https://github.com/garrettdevereux/ComputerVisionFinal/blob/main/Results/TrainingResults/resnetv2Training.txt">Training Loss</a> ended at 0.015. <b>94.6% Testing Accuracy after 20 epochs</b>. <a href="https://github.com/garrettdevereux/ComputerVisionFinal/blob/main/sportsImages/models/resnetv20Preds.csv"> Predictions versus Actual labels</a>.</p>

<p><b> EffNet: </b></p>
<p> 20 Epochs, Learning Rate schedule: {0:0.1, 5:0.01, 15: 0.001}, Batch Size: 128.</p>
<p> <img src="Results/TrainingResults/effnetTraining.png" alt="effnetTraining.png"></p>
<p> Performance: <a href="https://github.com/garrettdevereux/ComputerVisionFinal/blob/main/Results/TrainingResults/effnetTraining.txt">Training Loss</a> ended at 0.018. 95.6% Testing Accuracy after 20 epochs. <b>95.8% accuracy after 17 epochs</b>. <a href="https://github.com/garrettdevereux/ComputerVisionFinal/blob/main/sportsImages/models/effnet17Preds.csv"> Predictions versus Actual labels</a>.</p>

<p>Overall, the SimpleNet achieved 20.2% testing accuracy, the ConvNet hit 67.0%, ResNetv1 was at 92.2%, ResNetv2 increased to 94.6%, and EffNet ended at 95.8% accuracy.</p>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<h2> <img src="website/compare.png" alt="compare.png" width="50px" height="50px"> Comparison</h2>
<p><b> Validation Comparison </b></p>
<p>After spending all of this time training the models, I wanted to see what each model learned by comparing and contrasting how each one interpreted different images. By checking this, we could see if the models generally agreed, or if they failed how similar their incorrect prediction was to the correct one. I also thought it would be interesting to see if the pretrained models would ever disagree. Here are the results I found when running each model on a random batch of validation photos: </p>
<p> 
<img src="Results/ComparisonResults/validImage1.png" alt="c1.png" width="300px">
<img src="Results/ComparisonResults/validImage2.png" alt="c2.png" width="300px">
<img src="Results/ComparisonResults/validImage3.png" alt="c3.png" width="300px">
</p>
<p>
<img src="Results/ComparisonResults/validImage4.png" alt="c4.png" width="300px">
<img src="Results/ComparisonResults/validImage5.png" alt="c5.png" width="300px">
<img src="Results/ComparisonResults/validImage7.png" alt="c7.png" width="300px">
</p>
<p>
<img src="Results/ComparisonResults/validImage8.png" alt="c8.png" width="300px">
<img src="Results/ComparisonResults/validImage9.png" alt="c9.png" width="300px">
<img src="Results/ComparisonResults/validImage11.png" alt="c11.png" width="300px">
</p>

<p> Overall, these results were very consistent with the testing accuracy. SimpleNet only got two out of the nine correct, identifying horse racing and basketball. Every other model was also able to correctly identify these images. For the images it missed, it seemed to have only learned background colors, such as its choice of water polo for pole vault or figure skating men for canoe slalom. The ConvNet was also consistent with the testing accuracy, getting six out of nine correct. For the majority of the images, the trend seemed to be the three pretrained models getting it right, and ConvNet being hit or miss. For example, it is able to correctly identify many that SimpleNet failed at such as bowling, pole vault, and canoe slalom but still makes some poor guesses like judo for shuffleboard and bungee jumping for golf. Next, the three pretrained models all achieved similar results. Interestingly, none of the models were able to get the first photo of horseshoe pitching. The frisbee guess makes a lot of sense, but it goes to show the models are not perfect. Additionally, these models disagree on the pole vault photo, where RestNetv2 guesses high jump. This is a very small mistake as the sports are very close, but it is interesting to see that each model learned different features. </p>

<p><b> Real World Comparison </b></p>
Finally, I wanted to see how the models could do off in the real world. I compiled the set of images realworld which contains 20 images of me doing a bunch of different activities. These include photos of me playing football and baseball in high school (and even T-ball), kayaking, paddle boarding and surfing, and other images of me and my friends posing for a photo on a basketball court, football field, or golf course. These photos have no labels, and some don't even match a sport, but we can still examine how the models make their predictions. Here are the results: </p>

<p> 
<img src="Results/ComparisonResults/realImage1.png" alt="c1.png" width="300px">
<img src="Results/ComparisonResults/realImage4.png" alt="c4.png" width="370px">
<img src="Results/ComparisonResults/realImage2.png" alt="c2.png" width="300px">
</p>
<p>
<img src="Results/ComparisonResults/realImage3.png" alt="c3.png" width="300px">
<img src="Results/ComparisonResults/realImage10.png" alt="c10.png" width="370px">
<img src="Results/ComparisonResults/realImage6.png" alt="c6.png" width="270px">
</p>
<p>
<img src="Results/ComparisonResults/realImage7.png" alt="c7.png" width="300px">
<img src="Results/ComparisonResults/realImage18.png" alt="c18.png" width="300px">
<img src="Results/ComparisonResults/realImage14.png" alt="c14.png" width="300px">
</p>
<p>
<img src="Results/ComparisonResults/realImage13.png" alt="c13.png" width="310px">
<img src="Results/ComparisonResults/realImage11.png" alt="c11.png" width="300px">
<img src="Results/ComparisonResults/realImage20.png" alt="c20.png" width="300px">
</p>
<p>
<img src="Results/ComparisonResults/realImage12.png" alt="c12.png" width="300px">
<img src="Results/ComparisonResults/realImage19.png" alt="c19.png" width="310px">
<img src="Results/ComparisonResults/realImage15.png" alt="c15.png" width="310px">
</p>
<p>
<img src="Results/ComparisonResults/realImage16.png" alt="c16.png" width="300px">
<img src="Results/ComparisonResults/realImage5.png" alt="c5.png" width="370px">
<img src="Results/ComparisonResults/realImage17.png" alt="c17.png" width="270px">
</p>
<p>
<img src="Results/ComparisonResults/realImage9.png" alt="c9.png" width="400px">
<img src="Results/ComparisonResults/realImage8.png" alt="c8.png" width="420px">
</p>


<p>As these images are much different from the training and testing sets, it makes sense that the models don't do great. It seems that most of the action shots are classified pretty well by the models, but there is a lot of confusion for the photos where I am just standing and posing. Overall, it is really interesting to see how each model's learning transfers to a different set in the real world.</p>

<p>Note: See the 'Comparison of the Models', 'Comparison Function', 'Compare on Validation', and 'Real World Application' in <a href="https://github.com/garrettdevereux/ComputerVisionFinal/blob/main/Final_Project.ipynb">Final_Project.ipynb</a> for a full explanation as well as how the code was implemented to aquire these results.</p>


![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<h2> <img src="website/references.png" alt="ref.png" width="50px" height="50px">  References</h2>
<ul>
  <li><p>Joseph Redmon, 'iPynb Tutorial Series'. [Online].</p>
      <p>Available: <a href = "https://courses.cs.washington.edu/courses/cse455/22sp/">https://courses.cs.washington.edu/courses/cse455/22sp/</a></p>
  </li>
  <li><p>Gerry's Kaggle Dataset, '100 Sports Image Classification'. [Online].</p>
      <p>Available: <a href = "https://www.kaggle.com/datasets/gpiosenka/sports-classification?resource=download">https://www.kaggle.com/datasets/gpiosenka/sports-classification?resource=download</a></p>
  </li>
</ul>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<!-- CREDITS -->
<h2 id="credits"> Credits</h2>

Garrett Devereux

[![GitHub Badge](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/garrettdevereux)
[![LinkedIn Badge](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/garrett-devereux/)
