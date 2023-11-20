Overview
----
This repository is an implementation of the paper "[An Effective Deep Learning Adversarial Defense Method based on Spatial Structural Constraints in Embedding Space]".

Introduction
----
This paper proposes an effective deep learning adversarial defense method, which incorporates information about the spatial structures of the natural and the adversarial samples in the embedding space during the training process.


Predictive Behavior of CNN on Adversarial Samples
----
<p><img src="Figures/Motivation.PNG" alt="test" width="800"></p

Empirical investigation on the predictive behavior of CNN on adversarial samples from CIFAR-10 and CIFAR-100. The line (black, right y-axis) represents the number of increased successful attacks when perturbation is increased from its previous grid value. Each bar (left y-axis) represents the percentage of misclassification for the increased successful attacks, measuring number of adversarial samples are misclassified into the 2nd, 3rd, 4th and 5th most probable classes. FGSM and MIM are attack methods.
      

Demo of Training effect
----
<p><img src="Figures/Effect.PNG" alt="test" width="400"></p>

Here, we demonstrate the synergistic effect of the gaps at both probability and feature levels. As shown in the top, the average probability gaps between the true class and the most probable false class of ResNet-56 on CIFAR-100 testdata are 0.527 (CE loss) vs. 0.558 (PC loss). And for Tiny ImageNet test data the gaps become 0.131 (CE loss) vs. 0.231 (PC loss). These results demonstrate that our PC loss can directly enlarge the probability gap of prediction and the effect is more pronounced for more challenging dataset (Tiny ImageNet). As shown in the bottom, ResNet-56 trained with our PC loss has clear margin boundaries and samples of each classes are evenly distributed around the center with a minimal overlap on CIFAR-10 test data.


Results
----
<p><img src="Figures/MNIST.PNG" alt="test" width="800"></p>

T-SNE visualization of the penultimate layer of the model trained by CE loss (a,b) and our PC loss (c,d) on MNIST dataset. (a,c) display only clean images whereas (b,d) also include successful attacks generated with FGSM.
For a model trained with PC loss, due to the large margin between classes, the adversarial samples are harder to cross the boundaries with the only exception that the adversarial samples are distributed near the center of the feature space where hard samples are usually located.


Model Training and Evaluation
----

PC loss with logit constraints
* The PC loss is defined in "hard_margin_loss.py", while the Logit Constraints is defined in "margin_loss_soft_logit.py". As shown in the "vgg_training.py" file, by simply importing these two components and using them as a drop-in replacement of the CE loss, they can directly improve model's adversarial robustness for free. 

Other files
* The "vgg_training.py" shows how to use our method to do the training while "adv_testing.py" shows how to do the adversarial testing. The "models" folder contains all model architectures used in the experiments, which can be used to replace models in "vgg_training.py".

Note that the model needs warm-up with CE loss, and more training details can be found in our paper.


Dependencies
-----
* Python 3.11.4
* torch==2.1.0
* torchvision==0.16.0
* transformers==4.29.2
* pandas==1.5.3
* numpy==1.24.3
