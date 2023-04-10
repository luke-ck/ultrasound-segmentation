import gzip
import pickle
import re
import os
import cv2
import numpy as np


def process_image(image, target_size=(224, 224), is_mask=False):
    # Compute image dimensions
    h, w = image.shape[:2]

    # Compute padding required to create square aspect ratio
    pad_w = max(h - w, 0)
    pad_h = max(w - h, 0)

    # Pad image with black pixels
    padded_image = cv2.copyMakeBorder(image, top=pad_h // 2, bottom=pad_h - pad_h // 2, left=pad_w // 2,
                                      right=pad_w - pad_w // 2, borderType=cv2.BORDER_CONSTANT, value=0)

    if not is_mask:
        # Apply non-local means denoising
        denoised = cv2.fastNlMeansDenoising(padded_image, h=10)

        # Apply adaptive histogram equalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        padded_image = clahe.apply(denoised)

    # Resize image to target size
    resized = cv2.resize(padded_image, target_size, interpolation=cv2.INTER_LINEAR)

    return resized


def load_img(path, grayscale=False, target_size=None, is_mask=False):
    # if image does not exist return None
    if grayscale:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(path)
    if target_size:
        img = process_image(img, target_size, is_mask)
    # check if image is normalized
    if np.max(img) > 1:
        img = img / 255

    img = np.expand_dims(img, axis=-1)
    return img


def list_images(directory, ext='jpg|jpeg|bmp|png|tif'):
    return [os.path.join(directory, f) for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f)) and re.match('([\w]+\.(?:' + ext + '))', f)]


def stratified_bootstrap(y_train, frame_indexes):
    num_bootstrap_samples = y_train.shape[0] - len(frame_indexes)

    # perform bootstrapping
    bootstrap_mask_indexes = np.random.choice(frame_indexes, size=num_bootstrap_samples, replace=True)
    bootstrap_masks = y_train[bootstrap_mask_indexes]

    return bootstrap_masks, bootstrap_mask_indexes


class DataManager(object):
    DATA_PATH = './data/'

    TARGET_SIZE = (224, 224)

    @staticmethod
    def load_zipped_pickle(filename):
        with gzip.open(filename, 'rb') as f:
            loaded_object = pickle.load(f)
            return loaded_object

    @staticmethod
    def _read_preprocess_data(type='train'):
        data_path = os.path.join(DataManager.DATA_PATH, type)
        data = DataManager.load_zipped_pickle(data_path + '.pkl')
        print('Loaded {} data'.format(type))
        print('Starting preprocessing and bootstrapping...')
        total = 0
        for i, patient in enumerate(data):
            if type == 'train':
                video = [process_image(patient['video'][..., frame].astype(np.uint8), DataManager.TARGET_SIZE) for frame in
                     range(patient['video'].shape[-1])]
                mask = [
                    process_image(patient['label'][..., frame].astype(np.uint8), DataManager.TARGET_SIZE, is_mask=True)
                    for frame in
                    range(patient['label'].shape[-1])]
                mask = np.array(mask)

                video = np.array(video)

                video = video / 255.0

                bootstrapped_annotated_frames, bootstrap_mask_indexes = stratified_bootstrap(mask, patient['frames'])
                mask = np.concatenate((mask, bootstrapped_annotated_frames), axis=0).astype(np.float32)
                video = np.concatenate((video, video[bootstrap_mask_indexes]), axis=0).astype(np.float32)
                patient['video'] = video
                patient['label'] = mask

            else:
                video = [process_image(patient['video'][..., frame].astype(np.uint8), DataManager.TARGET_SIZE) for frame in
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
            f"Starting train/val split. Total expert frames: {expert_total_frames}, total amateur frames: {amateur_total_frames}")
        # split ratio is 80% train, 20% validation
        # greedily split the frames into train and validation going through each patient until the split ratio is reached
        # if the cumulative number of frames is less than the split ratio, then add the patient to the train set

        amateur_train_images, amateur_train_masks, amateur_val_images, amateur_val_masks = self.train_val_split(
            amateur_total_frames, amateur_train_data)
        expert_train_images, expert_train_masks, expert_val_images, expert_val_masks = self.train_val_split(
            expert_total_frames, train_data)

        print(f"Finished train/val split.")

        print('Saving {} .npy files...'.format(name_prefix))
        DataManager.save_train_np_data(amateur_train_images, amateur_train_masks, amateur_val_images, amateur_val_masks,
                                       name_prefix='amateur')
        DataManager.save_train_np_data(expert_train_images, expert_train_masks, expert_val_images, expert_val_masks,
                                       name_prefix='expert')

        print('Saving {} .npy files done.'.format(name_prefix))
        # by default the split will be balanced since we bootstrapped the data before.
        return (expert_train_images, expert_train_masks, expert_val_images, expert_val_masks), (
            amateur_train_images, amateur_train_masks, amateur_val_images, amateur_val_masks)

    def create_test_data(self, name_prefix='test'):
        test_data = self._read_preprocess_data(type=name_prefix)
        print('Saving {} .npy files...'.format(name_prefix))
        DataManager.save_np_arr(test_data, name_prefix=name_prefix)
        print('Saving {} .npy files done.'.format(name_prefix))
        return test_data

    def train_val_split(self, amateur_total_frames, amateur_train_data):
        train_images = []
        train_masks = []
        train_frames = 0
        val_images = []
        val_masks = []

        for patient in amateur_train_data:
            if train_frames / amateur_total_frames < 0.8:
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

    @staticmethod
    def save_train_np_data(X_train, y_train, X_val, y_val, name_prefix):
        DataManager.save_np_arr(X_train, name_prefix='{}_X_train'.format(name_prefix))
        DataManager.save_np_arr(y_train, name_prefix='{}_y_train'.format(name_prefix))
        DataManager.save_np_arr(X_val, name_prefix='{}_X_val'.format(name_prefix))
        DataManager.save_np_arr(y_val, name_prefix='{}_y_val'.format(name_prefix))

    @staticmethod
    def save_np_arr(arr, name_prefix):
        np.save(os.path.join(DataManager.DATA_PATH, name_prefix + '.npy'), arr)

    @staticmethod
    def load_np_arr(name):
        return np.load(os.path.join(DataManager.DATA_PATH, name))

    @staticmethod
    def load_train_val_data(name_prefix):
        X_train = DataManager.load_np_arr('{}_X_train.npy'.format(name_prefix))
        X_val = DataManager.load_np_arr('{}_X_val.npy'.format(name_prefix))
        y_train = DataManager.load_np_arr('{}_y_train.npy'.format(name_prefix))
        y_val = DataManager.load_np_arr('{}_y_val.npy'.format(name_prefix))
        X_train = np.expand_dims(X_train, axis=1)
        X_val = np.expand_dims(X_val, axis=1)
        y_train = np.expand_dims(y_train, axis=1)
        y_val = np.expand_dims(y_val, axis=1)
        return X_train, X_val, y_train, y_val

    @staticmethod
    def load_test_data():
        return DataManager.load_np_arr('test.npy')
