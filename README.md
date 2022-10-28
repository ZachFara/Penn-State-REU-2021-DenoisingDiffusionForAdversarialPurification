# Penn-State-REU-2021-DenoisingDiffusionForAdversarialPurification
This is all of the code that I found and created for my project in the 2021 Penn State REU on Machine Learning+Security. This project was overseen by Dr. Jinghui Chen.

Please note this project is a work in progress. Not all of the code functions. However, if you read the code on the attacks, DDPM, and CertifiedRobustnessModel it should all work. You will have to run the attacks on the resnet on your local computer, this code is unavailable because the attacks were not conducted in this directory. Also the DenoisingDiffusion models were not trained in this directory.

The ongoing experiments are present in TestingModels.ipynb

The goals of this project were to improve upon existing denoising diffusion models. 

1. In this project we did this through adding different noise distributions to a pertubed image. These images were then denoised, classified, and the results were added in a soft-voting manner.

2. The second improvement we developed was to certified robustness models. In one step of a certified robustness model, you add noise to an image, denoise it, then classify it. You repeat this 1000 times and add the classifications in a hard voting manner. The improvement we made was to use a surrogate model relating the variance of the noise added to the chance of successful classification. This is effectively bayesian optimization using a gaussian process regressor. This improved the results at a cost to computation cost.


Citations:

Carlini, N., Tramer, F., Krishnamurthy, Dvijotham, &amp; Kolter, J. Z. (2022, June 21). (certified!!) adversarial robustness for free! arXiv.org. Retrieved October 28, 2022, from https://arxiv.org/abs/2206.10550 

Ho, J., Jain, A., &amp; Abbeel, P. (2020, December 16). Denoising Diffusion Probabilistic models. arXiv.org. Retrieved October 28, 2022, from https://arxiv.org/abs/2006.11239 
