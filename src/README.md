# Different scripts to make image segmentation work

- basicblock.py & drunet.py - main files for the denoising model
- model.py - main file for our segmentation model
- transforms.py - file for data augmentation transforms. These are custom transforms that take in both the image and the mask and apply the same transform (deterministically) to both.
- data_manager.py - loads the data and applies the preprocessing steps and bootstrapping
- misc.py - contains training and validation logic
- utils.py - contains helper functions for running the model and plotting
- losses.py - contains custom loss functions