# Attacking Deep Learning with Adversarial Examples #

## Introduction to Adversarial Examples ##
AI technology based on deep learning is widely used today, which is very powerful tending to bring change to every aspects of our sociecty. Is there any method which can attack (or fool) our AI technology? actually, hackers can use adversarial examples to attack our AI system.  The adversarial examples is a input image which is intentionally designed to fool our models, but does not significantly interfere with the human eye. Specifically, adding imperceptible perturbations to the original input iamge does not cause any interference to the human eye, but DL model misclassify it as another object with high confidence. Imagine if the autonomous driving systems classify the red light sign as a bird (hacker's malicious attack), it could easily lead to a traffic accident.
## Minimum Requirements ##
 Requirement 1: Design at least two networks  
 Requirement 2: Train each net from two different initial weights.       
 Requirement 3: Generate many adversarial examples for each model and test them on all the models.  
 Requirement 4: Try a different set of epsilons and report its effects.    
 ## Dependencies ##

> * Python 3.7.3
> * NVIDIA GeForce GTX 1080
> * NVIDIA GeForce GTX Titan X
> * PyTorch 1.0.1
## Section 1: Adversarial Examples on MNIST ##
### FGSM Attack  ###
The fast gradient sign method, known as FGSM, was described by Goodfellow et. al. in [Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572). FGSM works well by using the gradients of the neural network to create an adversarial example. For an input image, the method uses the gradients of the loss with respect to the input image to create a new image which maximises the loss. This new image is called the adversarial image. 

```

def fgsm_attack(image, epsilon, data_grad):

    sign_data_grad = data_grad.sign()
    
    perturbed_image = image + epsilon*sign_data_grad
    
    return perturbed_image
    
```
### Statistical Results for  Requirement 1~3 ###

epsilon=0.05

|accurcy           | ResNet18_30 | ResNet18_60 |MobileNetV2_30 |MobileNetV2_60     | 
|:----------------:|:-----------:|:-----------:|:-------------:|:-----------------:|
|  ResNet18_30     | 41.27%      | 65.42%      |  88.09%       |   90.77%          |
|  ResNet18_60     | 60.96%      | 40.28%      |  87.67%       |   90.48%          |
|  MobileNetV2_30  | 91.09%      | 90.57%      |  11.99%       |   59.66%          |
|  MobileNetV2_60  | 91.22%      | 90.46%      |  33.70%       |   21.60%          |


epsilon=0.1

|accurcy           | ResNet18_30 | ResNet18_60 |MobileNetV2_30 |MobileNetV2_60     | 
|:----------------:|:-----------:|:-----------:|:-------------:|:-----------------:|
|  ResNet18_30     |14.75%       | 38.92%      |  52.69%       | 72.17%            |
|  ResNet18_60     |22.31%       | 19.39%      |  51.85%       | 70.13%            |
|  MobileNetV2_30  |74.24%       | 86.92%      |  10.71%       | 18.72%            |
|  MobileNetV2_60  |74.58%       | 86.98%      |               |  9.60%            |

FGSM uses the gradients of the loss with respect to the input image to create a new image which maximises the loss for the specific net, it works very well only on this net but not the others.

### Statistical Results for Requirement 4 ###

|nets            |epochs       |   epsilon    |accurcy after attack| nets        |epochs       |   epsilon    | accurcy after attack| 
|:--------------:|:-----------:|:------------:|:---------------:|:--------------:|:-----------:|:------------:|:---------------:|
|  ResNet18      |30           |0             |94.06%           | MobileNetV2    |30           |0             |93.56%           |
|  ResNet18      |30           |0.05          |41.27%           | MobileNetV2    |30           |0.05          |11.99%           |
|  ResNet18      |30           |0.1           |14.75%           | MobileNetV2    |30           |0.1           |10.71%           |
|  ResNet18      |30           |0.2           |8.87%            | MobileNetV2    |30           |0.2           | 8.60%           |
|  ResNet18      |30           |0.4           |8.83%            | MobileNetV2    |30           |0.4           | 8.60%           |
|  ResNet18      |30           |0.6           |9.14%            | MobileNetV2    |30           |0.6           | 8.60%           |
|  ResNet18      |60           |0             |91.86%           | MobileNetV2    |30           |0             |95.03%           |
|  ResNet18      |60           |0.05          |40.28%           | MobileNetV2    |60           |0.05          |21.60%           |
|  ResNet18      |60           |0.1           |19.39%           | MobileNetV2    |60           |0.1           |9.60%            |
|  ResNet18      |60           |0.2           |6.36%            | MobileNetV2    |60           |0.2           |8.47%            |
|  ResNet18      |60           |0.4           |3.40%            | MobileNetV2    |60           |0.4           |8.47%            |
|  ResNet18      |60           |0.6           |3.26%            | MobileNetV2    |60           |0.6           |8.47%            |



In our study, FGSM was only used on the right prediction cases, which means we ignore the original wrong prediction cases. As shown in the table, the accuracies decrease as the epsilon value increases. Also, note that here ϵ=0 case represents the original test accuracy on MNIST without any attack. All the pre-trained models are located [here]( https://drive.google.com/open?id=1FcU-uBOzDVu-J5Ag8Mr3iX2S0JdnE-i-).




