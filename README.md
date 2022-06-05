# ComputerVisionFinal
A Sports Images classifier experiment for UW's CSE455 Final

Overview/Excerpt:
All new data. All of the code is new to this project, not from any of the previous
homeworks done throughout the quarter. A lot of inspiration was taken from the 
tutorial series in Collab, but I have worked to tweak and add many unique aspects
for this project. I developed this project in google drive and not on github,
so many of the data processing steps in the begining will likely have to be 
tweaked in order to load the data on your own machine. explain file structure where everything is.

Issues with CPU/GPU

Problem Setup with Dataset:

We have already seen the simple model in several assignments such as hw5, 
and the darkNet is an optimization from one of the tutorials, but I worked
to train these networks for hours several times each to hone in learning rate
and other parameters.

Techniques:

Experimented with data augmentation. Found you can have too much
Resenet only getting 37% possibly compare with no aug
SimpleNet

DarkNet

Pretrain Resenet

Pretrain Resnet

Resnet 2: no rotations, no blur, do do normalization. A lot better loss, worse test.
Seems that augmentation did help overfitting.

EfficientNet: augmentations stats


Comparison between all:

Validation:

Realworld set:

Got kicked off GPU quick and spent lots of time waiting for training. 
Really cool to figure out how to do everything on my own, I feel that
I could confidently complete my own independent project.
