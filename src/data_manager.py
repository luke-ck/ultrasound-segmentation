import gzip
import pickle
import torch
import os
from typing import Optional, Tuple
from skimage import filters
from src.drunet import UNetRes
import cv2
import numpy as np
from src.utils import uint2single, single2tensor4, test_onesplit, tensor2uint, heal_image, stratified_bootstrap


class DataManager(object):

    def __init__(self, model_path: str, target_size: Optional[Tuple[int, int]] = None, device: str = 'cpu'):
        noise_level_model = 5
        self.noise_level_model_tensor = torch.FloatTensor([noise_level_model / 255.])
        self.TARGET_SIZE = target_size
        self.DATA_PATH = './data/'
        self.clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(4, 4))
        # Define kernel for morphology operations
        self.kernel = np.ones((2, 2), np.uint8)

        n_channels = 1
        # Load the pre-trained DRUNet model
        model_path = os.path.join(model_path, 'drunet_gray.pth')
        self.device = torch.device(device)
        torch.cuda.empty_cache()

        self.model = UNetRes(in_nc=n_channels + 1,
                             out_nc=n_channels,
                             nc=[64, 128, 256, 512],
                             nb=4,
                             act_mode='R',
                             downsample_mode="strideconv",
                             upsample_mode="convtranspose"
                             )
        self.model.load_state_dict(torch.load(model_path), strict=True)

        self.model.eval()
        for k, v in self.model.named_parameters():
            v.requires_grad = False
        self.model = self.model.to(device)

    @staticmethod
    def load_zipped_pickle(filename):
        with gzip.open(filename, 'rb') as f:
            loaded_object = pickle.load(f)
            return loaded_object

    def pad_image(self, image):
        # Compute image dimensions
        h, w = image.shape

        # Compute padding required to create square aspect ratio
        target_size = (max(h, w), max(h, w))
        padded_image = cv2.copyMakeBorder(image, 0, target_size[0] - h, 0, target_size[1] - w, cv2.BORDER_CONSTANT,
                                          value=0)
        return padded_image

    def process_mask(self, mask, target_size=(224, 224)):
        assert len(mask.shape) == 2, 'Mask must be grayscale'
        # Pad image to create square aspect ratio
        padded_mask = self.pad_image(mask)

        # Apply bilateral filter
        padded_mask = cv2.bilateralFilter(padded_mask, d=9, sigmaColor=1, sigmaSpace=1)

        # Threshold the filtered image
        threshold_value = filters.threshold_otsu(padded_mask)
        padded_mask = (padded_mask > threshold_value * 2).astype(np.uint8) * 255  # Threshold

        # Dilate the filtered image
        padded_mask = cv2.dilate(padded_mask, self.kernel, iterations=1)

        # Apply Gaussian blur to the dilated and filtered image
        padded_mask = cv2.GaussianBlur(padded_mask, (3, 3), 2)
        padded_mask = (padded_mask > threshold_value).astype(np.uint8) * 255

        # Resize image to target size
        resized = cv2.resize(padded_mask, target_size, interpolation=cv2.INTER_LINEAR)

        return resized

    def process_image(self, image, target_size=(224, 224), is_test=False, denoise_before=False):
        assert len(image.shape) == 2, 'Image must be grayscale'
        # Pad image to create square aspect ratio
        padded_image = self.pad_image(image)
        if not is_test:
            padded_image = self.denoise_enhance_image(denoise_before, padded_image)

        # Resize image to target size
        resized = cv2.resize(padded_image, target_size, interpolation=cv2.INTER_LINEAR)

        return resized

    def denoise_enhance_image(self, before: bool, image: np.ndarray) -> np.ndarray:
        image = self.denoise_image(image) if before else image
        image = self.clahe.apply(image)
        image = heal_image(image)
        image = self.denoise_image(image) if not before else image
        return image

    def denoise_image(self, image: np.ndarray) -> np.ndarray:
        image = np.expand_dims(image, axis=2)
        image = uint2single(image)
        image = single2tensor4(image)
        image = torch.cat((image,
                           self.noise_level_model_tensor.repeat(1, 1, image.shape[2], image.shape[3])),
                          dim=1)
        padded_image = image.to(self.device)
        denoised_img = test_onesplit(self.model, padded_image, refield=32)
        denoised_img = tensor2uint(denoised_img)
        return denoised_img

    def _read_preprocess_data(self, type='train'):
        data_path = os.path.join(self.DATA_PATH, type)
        data = DataManager.load_zipped_pickle(data_path + '.pkl')
        print('Loaded {} data'.format(type))
        print('Starting preprocessing and bootstrapping...')
        total = 0
        for i, patient in enumerate(data):
            if type == 'train':
                video = [self.process_image(patient['video'][..., frame].astype(np.uint8), self.TARGET_SIZE,
                                            denoise_before=False) for frame
                         in
                         range(patient['video'].shape[-1])]
                # augmentation: denoising before and after
                video += [self.process_image(patient['video'][..., frame].astype(np.uint8), self.TARGET_SIZE,
                                             denoise_before=True) for frame
                          in
                          range(patient['video'].shape[-1])]
                mask = [
                           self.process_mask(patient['label'][..., frame].astype(np.uint8), self.TARGET_SIZE)
                           for frame in
                           range(patient['label'].shape[-1])] * 2

                mask = np.array(mask)
                video = np.array(video)

                video = video / 255.0

                bootstrapped_annotated_frames, bootstrap_mask_indexes = stratified_bootstrap(mask, patient['frames'])
                mask = np.concatenate((mask, bootstrapped_annotated_frames), axis=0).astype(np.float32)
                video = np.concatenate((video, video[bootstrap_mask_indexes]), axis=0).astype(np.float32)

                # double the video and mask
                video = np.concatenate((video, video), axis=0).astype(np.float32)
                mask = np.concatenate((mask, mask), axis=0).astype(np.float32)

                patient['video'] = video
                patient['label'] = mask

            else:
                video = [self.process_image(patient['video'][..., frame].astype(np.uint8), self.TARGET_SIZE,
                                            is_test=True,
                                            denoise_before=False) for
                         frame in
                         range(patient['video'].shape[-1])]
                video = np.array(video)
                video = video / 255.0
                patient['video'] = video
            total += video.shape[0]

            if i % 5 == 0:
                print(f'Done {i} patients. Total frames: {total}')
        print('Finished preprocessing and bootstrapping')
        return data

    def create_train_data(self, name_prefix='train'):
        train_data = self._read_preprocess_data(type=name_prefix)
        # split into amateur and expert
        amateur_train_data = [patient for patient in train_data if patient['dataset'] == 'amateur']
        train_data = [patient for patient in train_data if patient['dataset'] == 'expert']

        # split into train and validation
        # first compute the total number of frames for each dataset type
        expert_total_frames = 0
        amateur_total_frames = 0
        for patient in train_data:
            expert_total_frames += patient['video'].shape[0]
        for patient in amateur_train_data:
            amateur_total_frames += patient['video'].shape[0]

        print(
            f"Starting train/val split. Total expert frames: {expert_total_frames}, "
            f"total amateur frames: {amateur_total_frames}")
        # split ratio is 80% train, 20% validation greedily split the frames into train and validation going through
        # each patient until the split ratio is reached if the cumulative number of frames is less than the split
        # ratio, then add the patient to the train set

        expert_train_images, expert_train_masks, expert_val_images, expert_val_masks = self.train_val_split(
            expert_total_frames, train_data)

        amateur_train_images, amateur_train_masks, amateur_val_images, amateur_val_masks = self.train_val_split(
            amateur_total_frames, amateur_train_data)

        print('Saving {} .npy files...'.format(name_prefix))
        self.save_train_np_data(amateur_train_images, amateur_train_masks, amateur_val_images, amateur_val_masks,
                                name_prefix='amateur')
        self.save_train_np_data(expert_train_images, expert_train_masks, expert_val_images, expert_val_masks,
                                name_prefix='expert')
        print('Saving {} .npy files done.'.format(name_prefix))
        print(f"Finished train/val split.")
        # by default the split will be balanced since we bootstrapped the data before.
        return (expert_train_images, expert_train_masks, expert_val_images, expert_val_masks), (
            amateur_train_images, amateur_train_masks, amateur_val_images, amateur_val_masks)

    def create_test_data(self, name_prefix='test'):
        test_data = self._read_preprocess_data(type=name_prefix)
        print('Saving {} .npy files...'.format(name_prefix))
        self.save_np_arr(self.DATA_PATH, test_data, name_prefix=name_prefix)
        print('Saving {} .npy files done.'.format(name_prefix))
        return test_data

    def train_val_split(self, amateur_total_frames: int, amateur_train_data: list, split_ratio=0.8) -> (
            list, list, list, list):
        """
        This function splits the amateur data into train and validation sets. It does so by greedily splitting the
        frames into train and validation going through each patient until the split ratio is reached. If the cumulative
        number of frames is less than the split ratio, then add the patient to the train set.
        Args:
            amateur_total_frames: total number of frames in the amateur dataset
            amateur_train_data: amateur dataset
            split_ratio: ratio of train to validation data

        Returns:
            train_images, train_masks, val_images, val_masks
        """
        train_images = []
        train_masks = []
        train_frames = 0
        val_images = []
        val_masks = []

        for patient in amateur_train_data:
            if train_frames / amateur_total_frames < split_ratio:
                train_images.append(patient['video'])
                train_masks.append(patient['label'])
                train_frames += patient['video'].shape[0]
            else:
                val_images.append(patient['video'])
                val_masks.append(patient['label'])

        val_images = np.concatenate(val_images, axis=0)
        val_masks = np.concatenate(val_masks, axis=0)
        train_images = np.concatenate(train_images, axis=0)
        train_masks = np.concatenate(train_masks, axis=0)

        return train_images, train_masks, val_images, val_masks

    def save_train_np_data(self, X_train, y_train, X_val, y_val, name_prefix):
        self.save_np_arr(self.DATA_PATH, X_train, name_prefix='{}_X_train'.format(name_prefix))
        self.save_np_arr(self.DATA_PATH, y_train, name_prefix='{}_y_train'.format(name_prefix))
        self.save_np_arr(self.DATA_PATH, X_val, name_prefix='{}_X_val'.format(name_prefix))
        self.save_np_arr(self.DATA_PATH, y_val, name_prefix='{}_y_val'.format(name_prefix))

    @staticmethod
    def save_np_arr(path, arr, name_prefix):
        np.save(os.path.join(path, name_prefix + '.npy'), arr)

    @staticmethod
    def load_np_arr(path, name):
        return np.load(os.path.join(path, name))

    def load_train_val_data(self, name_prefix):
        X_train = DataManager.load_np_arr(self.DATA_PATH, '{}_X_train.npy'.format(name_prefix))
        X_val = DataManager.load_np_arr(self.DATA_PATH, '{}_X_val.npy'.format(name_prefix))
        y_train = DataManager.load_np_arr(self.DATA_PATH, '{}_y_train.npy'.format(name_prefix))
        y_val = DataManager.load_np_arr(self.DATA_PATH, '{}_y_val.npy'.format(name_prefix))
        X_train = np.expand_dims(X_train, axis=1)
        X_val = np.expand_dims(X_val, axis=1)
        y_train = np.expand_dims(y_train, axis=1)
        y_val = np.expand_dims(y_val, axis=1)
        return X_train, X_val, y_train, y_val

    def load_test_data(self):
        return DataManager.load_np_arr(self.DATA_PATH, 'test.npy')
