# Using deep learning to automatically break captchas
## What are captchas?
Completely Automated Public Turing test to tell Computers and Humans Apart (CAPTCHA) is a way of differentiating humans and machines and was coined by von Ahn, Blum, Hopper, and Langford [5]. The core idea is that reading distorted letters, numbers, or images is achievable for a human but very hard or impossible for a computer. A simple captcha can look like the one here:
![simple captcha](pics/Penguin-Pal_Captcha.png)

There are several use cases for captchas, which includes the ones presented in [6]:
- Preventing comment spam
- Protect website registration
- Protect e-mail addresses from scrappers
- Online polls
- Preventing Dictionary Attacks
- Search Engine Bots

There are however attack vectors to break captcha system. These include cheap or unwitting human labor, insure implementation, and machine learning based attacks.

Captchas are based on an unsolved AI problem. However, with the progress of AI techniques and computing power, captchas can be broken as shown in [2], [3], and [4].

![captcha](pics/red-captcha.png)

Google recaptcha

![recaptcha](pics/noCaptcha-mobile.png.gif)

## Our objectives and motivation
The aim of the project is to break captchas using deep learning technologies.
Initially we focus on simple captchas to evaluate the performance and move into complex captchas. The training dataset will be generated from an open source captcha generation software. Tensorflow will be used to create and train a neural network.

## Creating the datasets


## A naive approach to captcha breaking



## TBD: Our further things

## Conclusion


## References
1. Goodfellow, Ian J., et al. "Multi-digit number recognition from street view imagery using deep convolutional neural networks." arXiv preprint arXiv:1312.6082 (2013).
2. Karthik, Colin Hong Bokil Lopez-Pineda, and Rajendran Adria Recasens. "Breaking Microsoft’s CAPTCHA." (2015).
3. "Using deep learning to break a Captcha system | Deep Learning." 3 Jan. 2016, https://deepmlblog.wordpress.com/2016/01/03/how-to-break-a-captcha-system/. Accessed 6 Dec. 2016.
4. Stark, Fabian, et al. "CAPTCHA Recognition with Active Deep Learning." Workshop New Challenges in Neural Computation 2015. 2015.
5. Von Ahn, Luis, et al. "CAPTCHA: Using hard AI problems for security." International Conference on the Theory and Applications of Cryptographic Techniques. Springer Berlin Heidelberg, 2003.
6. "CAPTCHA: Telling Humans and Computers Apart Automatically" 2010, http://www.captcha.net/. Accessed 7 Jan. 2017.
