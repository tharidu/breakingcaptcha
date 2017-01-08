# Using deep learning to automatically break captchas
## Introduction
Completely Automated Public Turing test to tell Computers and Humans Apart (CAPTCHA) is a way of differentiating humans and machines and was coined by von Ahn, Blum, Hopper, and Langford [5]. The core idea is that reading distorted letters, numbers, or images is achievable for a human but very hard or impossible for a computer. A simple captcha can look like the two below:
![simple captcha](pics/Penguin-Pal_Captcha.png)
*Simple captcha with two different fonts and slight rotation*


![captcha](pics/red-captcha.png)
*reCAPTCHA example with two words, rotation and distortion*

There are several use cases for captchas, which includes the ones presented in [6]:
- Preventing comment spam
- Protect website registration
- Protect e-mail addresses from scrappers
- Online polls
- Preventing Dictionary Attacks
- Search Engine Bots

There are however attack vectors to break captcha system. These include cheap or unwitting human labor, insecure implementation, and machine learning based attacks. We will not go into detail on insecure implementations, as the focus of this article are deep learning based approaches.

### Human based captcha breaking
Out of curiosity we take a look at the human based approach. For example [BypassCaptcha](http://bypasscaptcha.com/order1.php) offers breaking captchas with cheap human labor in packages (e.g. 20,000 captchas for 130$). There are also other services including [Image Typerz](http://www.imagetyperz.com/Forms/bypasscaptcha.aspx), [ExpertDecoders](http://expertdecoders.com/), and [9kw.eu](https://www.9kw.eu/). There are also hybrid solutions that use both OCR and human labor like [DeathByCaptcha](http://deathbycaptcha.com/user/login). These vendors list the following accuracies and response times (averages):

| Service | Accuracy (daily average) | Response Time (daily average)|
|---------|---------:|--------------:|
| BypassCaptcha | N/A | N/A |
| Image Typerz | 95% | 10+ sec |
| ExpertDecoders | 85% | 12 sec |
| CaptchaBOSS (premium version of ExpertDecoders) | 99% | 8 sec |
| 9kw.eu | N/A | 30 sec |
| DeathByCaptcha | 96.8% | 10 sec |

The values are advertised and self-reported. We did not conduct any verification of the stated numbers, but it can give an orientation what a machine learning based approach should achieve.


### ML based captcha breaking
Captchas are based on an unsolved AI problem. However, with the progress of AI techniques and computing power, captchas can be broken as shown in [2], [3], and [4].

Google recaptcha

![recaptcha](pics/noCaptcha-mobile.png.gif)

## Our objectives and motivation
The aim of the project is to break captchas using deep learning technologies. Initially we focus on simple captchas to evaluate the performance and move into complex captchas. The training dataset will be generated from an open source captcha generation software. Tensorflow will be used to create and train a neural network.

## Existing research

## Creating the datasets
We are generating the datasets


## A naive approach to captcha breaking
As a first step we use quite simple captchas as displayed below.
![simplegenerated](pics/rgy8a.jpg)



## TBD: Our further things

## Conclusion


## References
1. Goodfellow, Ian J., et al. "Multi-digit number recognition from street view imagery using deep convolutional neural networks." arXiv preprint arXiv:1312.6082 (2013).
2. Karthik, Colin Hong Bokil Lopez-Pineda, and Rajendran Adria Recasens. "Breaking Microsoft’s CAPTCHA." (2015).
3. "Using deep learning to break a Captcha system | Deep Learning." 3 Jan. 2016, https://deepmlblog.wordpress.com/2016/01/03/how-to-break-a-captcha-system/. Accessed 6 Dec. 2016.
4. Stark, Fabian, et al. "CAPTCHA Recognition with Active Deep Learning." Workshop New Challenges in Neural Computation 2015. 2015.
5. Von Ahn, Luis, et al. "CAPTCHA: Using hard AI problems for security." International Conference on the Theory and Applications of Cryptographic Techniques. Springer Berlin Heidelberg, 2003.
6. "CAPTCHA: Telling Humans and Computers Apart Automatically" 2010, http://www.captcha.net/. Accessed 7 Jan. 2017.
