# Breaking captcha repository
This repository is used to track the report and code for our project in Scalable Machine Learning and Deep Learning (ID2223) at KTH.

The aim of the project is to break captchas using deep learning technologies. Initially we will focus on simple captchas to evaluate the performance and move into more complex captchas. The training dataset will be generated from an open source captcha generation software. Tensorflow will be used to create, train and test the network.

## Generating the dataset
The datasets is generated using a Java based captcha generator. The easiest way is to import the Java project into IntelliJ, compile, and execute  `breakingcaptcha/data_gen/src/kth/id2223/Main.java`. We have generated the following datasets.

| Description | Size | Training samples | Test samples |
|:------------|-----:|-----------------:|-------------:|
| Digits only | 38 MB | 9502 | 100 |
| Digits and characters | 197 MB | 49796 | 100 |
| Digits and characters with rotation | 39 MB | 10000 | 100 |
| Digits and characters with rotation | 198 MB | 49782 | 500 |
| Digits and characters with rotation | 777 MB | 196926 | 500 |

Each dataset contains jpeg images containing a captcha with five characters. The characters are lowercase letters (a-z) or numbers (0-9). We used the fonts "Arial" and "Courier" with and without rotation. An example of the created captchas is displayed below. Our intention was to mimic the captchas created by [Microsoft](https://courses.csail.mit.edu/6.857/2015/files/hong-lopezpineda-rajendran-recansens.pdf).

![Captcha1](report/pics/54563.jpg) *Simple digit-only captcha* </br>
![Captcha2](report/pics/5p23r.jpg) *Characters and digits without rotation* </br>
![Captcha3](report/pics/ycn2m.jpg) *Characters and digits with rotation*


## Executing the CNN
We have developed the code using Tensorflow 8.0 with Python 2.7.6 and Nvidia GPUs with CUDA 8.0. The CNN can be executed from the root folder of the project with
`cd simple_CNN` and
`PYTHONPATH=".." python captcha_cnn.py`.

After a successful training, the learned model is saved as `model.ckpt`.
Then you can use `PYTHONPATH=".." python restore_captcha_cnn.py filename.jpg` to predict an image file using the learned model.

## Results
#### First unsuccessful tries
![DigitsOnly660M](report/pics/digits_only_660M.png) </br>
*CNN with three conv. layers and two fully connected layers accuracy of CAPTCHAs with 5 digits or lowercase letters without rotation. Training in 100 batches and 10000 training samples.*

#### Digit-only captchas
![DigitsOnly](report/pics/digits_only.png) </br>
*CNN with three conv. layers and two fully connected layers accuracy of CAPTCHAs with five digits without rotation. Training in 157 batches, 39250 training samples, and testing with 100 CAPTCHAs.*

#### Digit and letter captchas
![DigitsChar](report/pics/digits_char.png) </br>
*CNN with three conv. layers and two fully connected layers accuracy of CAPTCHAs with five digits or lowercase letters without rotation. Training in 199 batches, 49750 training samples, and testing with 500 CAPTCHAs.*


#### Digit and letter captchas with rotation
![DigitsCharRot](report/pics/digits_char.png) </br>
*CNN with three conv. layers and two fully connected layers accuracy of CAPTCHAs with five digits or lowercase letters with rotation. Training in 787 batches, 196926 training samples, and testing with 500 CAPTCHAs.*

#### Examples of correct and false predictions
| Captcha  | Prediction |  Captcha  | Prediction |
|:-------:|:-|:-----:|:-|
| ![correct1](report/pics/54563.jpg) | 54563 | ![false1](report/pics/82290.jpg) | 8229**8** |
| ![correct2](report/pics/grh56.jpg) | grh56 | ![false2](report/pics/h76ap.jpg) | **k**76ap |
| ![correct3](report/pics/fb2x4.jpg) | fb2x4 | ![false3](report/pics/fffgf.jpg) | fffg**r** |






## Report
We have compiled a full blog-style [report](report/CAPTCHA-report.md) with more details.
