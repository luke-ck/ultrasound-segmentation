

# I've taken a look around the web to see what they do in competitions, and compiled a list of resources:

1. https://github.com/dorltcheng/Transfer-Learning-U-Net-Deep-Learning-for-Lung-Ultrasound-Segmentation
2. https://www.kaggle.com/micajoumathematics/my-first-semantic-segmentation-keras-u-net#Exploratory-data-analysis
3. https://www.kaggle.com/rakhlin/fast-run-length-encoding-python
4. https://www.kaggle.com/chefele/plot-images-overlaid-with-mask

# Blog and code about certain insights about the challenge (Dice score is similar to Jaccard) on similar segmentation:
- https://github.com/raghakot/ultrasound-nerve-segmentation
- https://raghakot.github.io/2016/12/26/Ultrasound-nerve-segmentation-challenge-on-Kaggle.html

Please read about common pitfalls using the IoU/Dice (especially when using U-net) and let's come up with some ideas on how to tackle this. Specifically, I think that if we take a DL approach, we need to be mindful of a couple of things:
1. We have the same problem regarding presence of segmentation mask as the blog says, namely:
> What makes this challenge particularly interesting is that ~60% of images do not contain the brachial plexus, i.e., segmentations mask is empty. Since the competition uses the mean of dice coefficient across all samples as the evaluation metric, even a single pixel in segmentation mask for non-nerve images mask kills the score. Ironically, you can achieve a score of ~0.6 just by predicting zero masks; it drops to ~[0.35, 0.45] when you try to learn it using a standard U-Net. This, combined with nosiy ultrasound images and a lack of obvious nerve patterns, intrigued me enough to participate in the challenge.

2. We need a gauge of how to create the bounding box of the mitral valve *because the IoU score varies wildly based on how aligned the predicted box is compared to base results*

<p align="center">
  <img width="300" height="234" src="https://upload.wikimedia.org/wikipedia/commons/c/c7/Intersection_over_Union_-_visual_equation.png">
  <img width="300" height="124" src="https://upload.wikimedia.org/wikipedia/commons/e/e6/Intersection_over_Union_-_poor%2C_good_and_excellent_score.png">
</p>

3. How do we score the *importance* of amateur vs professional labelled data i.e.
  * Do we use amateur labels at all?
  * Would we need to upsample the videos?
  * How would we weigh the predictions on amateur videos (since there is more uncertainty in the label)? 
  

# this is an imageGenerator for the data augmentation part:
- https://www.kaggle.com/hexietufts/easy-to-use-keras-imagedatagenerator
more data augmentation:
- https://www.kaggle.com/bguberfain/elastic-transform-for-data-augmentation/notebook

## And although this is for seismic ultrasounds I think it's worth looking at since it's very detailed and has a lot of helper functions:
- https://www.kaggle.com/ebberiginal/tgs-salt-keras-unet-depth-data-augm-strat

# overview of U-net:
1. https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/
2. https://arxiv.org/abs/2110.02196

there's a portable version of the model in the fastai package with some helper functions and etc. to get us started maybe:
- https://www.kaggle.com/tanlikesmath/ultrasound-nerve-segmentation-with-fastai



