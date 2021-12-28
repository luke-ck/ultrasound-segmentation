import os
import re
import cv2
import numpy as np

def load_img(path, grayscale=False, target_size=None):
    if grayscale:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(path)
    if target_size:
        img = cv2.resize(img, target_size)
    return img


def list_images(directory, ext='jpg|jpeg|bmp|png|tif'):
    return [os.path.join(directory, f) for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f)) and re.match('([\w]+\.(?:' + ext + '))', f)]

class DataManager(object):
    DATA_PATH = './data/'

    AM_IMG_ORIG_ROWS = 112     # Height
    AM_IMG_ORIG_COLS = 112     # Width

    # padding target size for expert dataset
    EX_IMG_TARGET_ROWS = 224 
    EX_IMG_TARGET_COLS = 224

    @staticmethod
    def read_train_images():
        train_data_path = os.path.join(DataManager.DATA_PATH, 'train')

        print('Loading training amateur images...')
        amateur_train_data_path = os.path.join(train_data_path, 'amateur')
        amateur_patient_classes, amateur_imgs, amateur_imgs_mask = DataManager._read_train_images(amateur_train_data_path)
        
        print('Loading training expert images...')
        
        expert_train_data_path = os.path.join(train_data_path, 'expert')
        expert_patient_classes, expert_imgs, expert_imgs_mask = DataManager._read_train_images(expert_train_data_path, True)
        
        return (amateur_patient_classes, amateur_imgs, amateur_imgs_mask), (expert_patient_classes, expert_imgs, expert_imgs_mask)

    #@staticmethod
    
    @staticmethod
    def _read_train_images(path, is_expert=False):
        """
        open the amateur/expert dirs and iteratively retrieve the saved images. In case of expert resize them to a set shape. This also retrieves
        the patient ids for the videos, transformed into integers
        """
        images = list_images(path)
        total = int(len(images) / 2)
        patient_classes = list()
        im_h = DataManager.AM_IMG_ORIG_ROWS
        im_w = DataManager.AM_IMG_ORIG_COLS
        
        if is_expert:
            im_h = DataManager.EX_IMG_TARGET_ROWS
            im_w = DataManager.EX_IMG_TARGET_COLS
            
        imgs = np.ndarray((total, im_h, im_w), dtype=np.uint8)
        imgs_mask = np.ndarray((total, im_h, im_w), dtype=np.uint8)
        
        i = 0
        for image_path in images:
            if 'mask' in image_path:
                continue

            image_name = os.path.basename(image_path)
            name = image_name.split('.')[0]
            patient_classes.append(name.split('_')[0])

            image_mask_name = name + '_mask.png'
            if is_expert:
                target_size = (im_h, im_w) # padding
            else:
                target_size = None
                
            imgs[i] = load_img(os.path.join(path, image_name), 
                               grayscale=True, 
                               target_size=target_size)
            imgs_mask[i] = load_img(os.path.join(path, image_mask_name), 
                                    grayscale=True, 
                                    target_size=target_size)

            if i % 100 == 0:
                print('Done: {0}/{1} images'.format(i, total))
            i += 1
        patient_classes = np.unique(patient_classes, return_inverse=True)[1] # convert labels into indexes
        return patient_classes, imgs, imgs_mask
    
    @staticmethod
    def create_train_data():
        amateur, expert = DataManager.read_train_images()
        am_patient_classes, am_imgs, am_imgs_mask = amateur
        ex_patient_classes, ex_imgs, ex_imgs_mask = expert
        
        print('Creating train dataset...')
        am_mask_labels = [1 if np.count_nonzero(mask) > 0 else 0 for mask in am_imgs_mask]
        ex_mask_labels = [1 if np.count_nonzero(mask) > 0 else 0 for mask in ex_imgs_mask]
        
        DataManager.save_train_val_split(am_imgs, am_imgs_mask, "amateur", stratify=am_mask_labels)
        DataManager.save_train_val_split(ex_imgs, ex_imgs_mask, "expert", stratify=ex_mask_labels)

    @staticmethod
    def create_test_data():
        train_data_path = os.path.join(DataManager.DATA_PATH, 'test')
        images = os.listdir(train_data_path)
        total = len(images)
        im_h = DataManager.EX_IMG_TARGET_ROWS # I guess we use only the expert recordings for now....
        im_w = DataManager.EX_IMG_TARGET_COLS
        imgs = np.ndarray((total, 1, im_h, im_w), dtype=np.uint8)
        imgs_id = list()

        print('Creating test images...')
        i = 0
        target_size = (im_h, im_w)
        for image_path in images:
            image_name = os.path.basename(image_path)
            imgs_id.append(image_name.split('.')[0])
            
            img = load_img(os.path.join(train_data_path, image_name), grayscale=True, target_size=target_size)
            img = np.array([img])
            imgs[i] = img

            if i % 100 == 0:
                print('Done: {0}/{1} images'.format(i, total))
            i += 1

        # Build all data set
        print('Saving test samples...')
        imgs_id = np.unique(imgs_id, return_inverse=True)[1] # convert labels into indexes
        imgs = imgs[np.argsort(imgs_id)]
        np.save(os.path.join(DataManager.DATA_PATH, 'imgs_test.npy'), imgs)
        print('Saving to .npy files done.')

    @staticmethod
    def save_train_val_split(X, y, name_prefix, stratify=None, split_ratio=0.1):
        X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=stratify, test_size=split_ratio)
        np.save(os.path.join(DataManager.DATA_PATH, '{}_X_train.npy'.format(name_prefix)), X_train)
        np.save(os.path.join(DataManager.DATA_PATH, '{}_X_val.npy'.format(name_prefix)), X_val)
        np.save(os.path.join(DataManager.DATA_PATH, '{}_y_train.npy'.format(name_prefix)), y_train)
        np.save(os.path.join(DataManager.DATA_PATH, '{}_y_val.npy'.format(name_prefix)), y_val)
        print('Saving {} .npy files done.'.format(name_prefix))


    @staticmethod
    def load_train_val_data(name_prefix):
        X_train = np.load(os.path.join(DataManager.DATA_PATH, '{}_X_train.npy'.format(name_prefix)))
        X_val = np.load(os.path.join(DataManager.DATA_PATH, '{}_X_val.npy'.format(name_prefix)))
        y_train = np.load(os.path.join(DataManager.DATA_PATH, '{}_y_train.npy'.format(name_prefix)))
        y_val = np.load(os.path.join(DataManager.DATA_PATH, '{}_y_val.npy'.format(name_prefix)))
        return X_train, X_val, y_train, y_val

    @staticmethod
    def load_test_data():
        return np.load(os.path.join(DataManager.DATA_PATH, 'imgs_test.npy'))
