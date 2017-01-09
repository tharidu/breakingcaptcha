# Breaking captcha repository
This repository is used to track the report and code for our project in Scalable Machine Learning and Deep Learning (ID2223) at KTH.

The aim of the project is to break captchas using deep learning technologies. Initially we will focus on simple captchas to evaluate the performance and move into complex captchas. The training dataset will be generated from an open source captcha generation software. Tensorflow will be used to train the network.

## Generating the dataset
The datasets can be generated using a Java based captcha generator. The easiest way is to import the Java project into IntelliJ, compile, and execute  `breakingcaptcha/data_gen/src/kth/id2223/Main.java`. We have generated the following datasets.
| Description | Size | Training samples | Test samples |
|:------------|-----:|-----------------:|-------------:|
| Digits only | 38 MB | 9502 | 100 |
| Digits and characters | 197 MB | 49796 | 100 |
| Digits and characters with rotation | 198 MB | 49782 | 500 |
| Digits and characters with rotation | 777 MB | 196926 | 500 |

Each dataset contains jpeg images containing a captcha with five characters. The characters are lowercase letters (a-z) or numbers (0-9). We used the fonts "Arial" and "Courier" with and without rotation. An example of the created captchas is displayed below. Our intention was to mimic the captchas created by [Microsoft](https://courses.csail.mit.edu/6.857/2015/files/hong-lopezpineda-rajendran-recansens.pdf).

![Captcha1](report/pics/54563.jpg) *Simple digit-only captcha* </br>
![Captcha2](report/pics/5p23r.jpg) *Characters and digits without rotation* </br>
![Captcha3](report/pics/ycn2m.jpg) *Characters and digits with rotation*


## Executing the CNN
We have developed the code using Tensorflow 8.0 with Python 2.7.6 and CUDA 8.0 with Nvidia drivers (361.93.02). The CNN can be executed from the root folder of the project with
`cd simple_CNN` and
`PYTHONPATH=".." python captcha_cnn.py`.

After a successful training, the learned model is saved as `model.ckpt`.
Then you can use `PYTHONPATH=".." python restore_captcha_cnn.py filename.jpg` to predict an image file using the learned model.

## Results
Digits only:
![DigitsOnly](report/pics/digits_only.png) </br>
*CNN with three conv. layers and two fully connected layers accuracy of captchas with 5 digits without rotation. Training with 157 batches, 39250 training samples, and testing with 100 captchas.*

## Report
We have compiled a full blog-style [report](report/CAPTCHA-report.md) with more details.
