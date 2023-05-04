import re
from pathlib import Path
from typing import Tuple, Any
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from PIL import Image
import gzip
import pickle
import torch
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
import torch.nn as nn
from src.data_generator import MyDataset
from src.data_manager import DataManager
from src.metrics import dice_coeff, iou_coeff, power_jaccard_loss
from src.transforms import Compose, Normalize, ConvertImageDtype, NumpyToTensor, RandomRotation, RandomAffine, Lambda, \
    GaussianBlur
from src.utils import plot_segmentation_overlay


def setup_dirs(path):
    # create directories for saving models and figures
    if not os.path.exists(path + '/checkpoints/'):
        os.mkdir(path + '/checkpoints/')
    if not os.path.exists(path + '/figures/'):
        os.mkdir(path + '/figures/')
        os.mkdir(path + '/figures/loss/')


def mask_not_blank(mask):
    return sum(mask.flatten()) > 0


def setup_optimization(model, learning_rate=3e-4, weight_decay=1e-8, momentum=0.999, amp=False):
    optimizer = torch.optim.RMSprop(model.parameters(),
                                    lr=learning_rate,
                                    weight_decay=weight_decay,
                                    momentum=momentum,
                                    foreach=True)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    return optimizer, scheduler, grad_scaler


def plot_image(img, title=None):
    plt.figure(figsize=(7, 7))
    plt.title(title)
    plt.imshow(img)


#     plt.show()

def load_zipped_pickle(filename):
    with gzip.open(filename, 'rb') as f:
        loaded_object = pickle.load(f)
        return loaded_object


def resize2SquareKeepingAspectRatio(img, size, interpolation):
    h, w = img.shape[:2]
    c = None if len(img.shape) < 3 else img.shape[2]
    if h == w:
        return cv2.resize(img, (size, size), interpolation)
    if h > w:
        dif = h
    else:
        dif = w
    x_pos = int((dif - w) / 2.)
    y_pos = int((dif - h) / 2.)
    if c is None:
        mask = np.zeros((dif, dif), dtype=img.dtype)
        mask[y_pos:y_pos + h, x_pos:x_pos + w] = img[:h, :w]
    else:
        mask = np.zeros((dif, dif, c), dtype=img.dtype)
        mask[y_pos:y_pos + h, x_pos:x_pos + w, :] = img[:h, :w, :]
    return cv2.resize(mask, (size, size), interpolation)


def grays_to_RGB(img):
    # turn image into grayscale RGB
    return np.array(Image.fromarray(img).convert("RGB"))


def save_img(img, img_idx, path, pid, is_mask=False):
    filename = path + '/' + str(pid) + '_' + str(img_idx)
    if is_mask:
        filename += '_mask.png'
        img = np.asarray(img, dtype="uint8")  # convert bool mask into uint8 so cv2 doesn't scream
    else:
        filename += '.png'
        img = grays_to_RGB(img)

    cv2.imwrite(filename, img)


def make_dir(path):
    try:
        os.mkdir(path)
    except OSError:
        print(f"Creation of the directory {path} failed", end='\r')


def gen_dataset(imgs, dataset, pid, labels=None, typeof_dataset=None):
    BASE = os.getcwd()
    output_dir = BASE + '/data/' + dataset + '/'
    if os.path.isdir(output_dir) is False:
        make_dir(output_dir)
    if typeof_dataset is not None:  # this is only for train
        output_dir += typeof_dataset  # + '/'
        if os.path.isdir(output_dir) is False:
            make_dir(output_dir)

    for i, img in enumerate(imgs):
        save_img(img, i, output_dir, pid)
        if labels is not None:  # this is only for train
            save_img(labels[i], i, output_dir, pid, is_mask=True)


def list_images(directory, ext='jpg|jpeg|bmp|png|tif'):
    return [os.path.join(directory, f) for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f)) and re.match('([\w]+\.(?:' + ext + '))', f)]


def torch_setup():
    print("GPU available: ", torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_default_dtype(torch.float32)
    return device


def plot_image_with_mask(img, mask, title=None):
    """
    this does the same thing as plot_separate but plots only the overlayed img
    """
    # returns a copy of the image with edges of the mask added in red
    img_mask = np.ma.masked_where(mask == False, mask)
    # img_box = np.ma.masked_where(box == False, box)
    plt.figure(figsize=(8, 8))
    plt.imshow(img, cmap='gray', interpolation='none')
    plt.imshow(img_mask, cmap='jet', interpolation='none', alpha=0.8)
    # plt.imshow(img_box, cmap = 'spring', interpolation = 'none', alpha = 0.5)
    plt.title(title)


@torch.inference_mode()
def evaluate(net, dataloader, device, epoch, amp, BASE, threshold=0.8, save_fig=False):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    iou_score = 0
    loss_fn = power_jaccard_loss
    loss = 0
    if save_fig:
        # save some predictions for visualization
        total_visualised_images = 0
        visualised_images = []

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred = net(image)

            assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
            # compute metrics
            loss += loss_fn(mask_pred, mask_true)

            mask_pred = (mask_pred > threshold).long()
            dice_score += dice_coeff(mask_pred, mask_true)
            iou_score += iou_coeff(mask_pred, mask_true)
            if save_fig:
                # find 5 indexes where the mask is not empty and save the image, mask and prediction
                for i in range(image.shape[0]):
                    # save some predictions for visualization
                    if mask_true[i, 0].sum() > 0 and total_visualised_images < 5:
                        total_visualised_images += 1
                        visualised_images.append((image[i, 0].detach().cpu().numpy(),
                                                  mask_true[i, 0].detach().cpu().numpy(),
                                                  mask_pred[i, 0].detach().cpu().numpy()))

    if save_fig:
        for i, (image, mask_true, mask_pred) in enumerate(visualised_images):
            # if i == 0:
            #     np.save(BASE + '/data/visualised_images/image_' + str(epoch) + '.npy', image)
            #     np.save(BASE + '/data/visualised_images/mask_true_' + str(epoch) + '.npy', mask_true)
            #     np.save(BASE + '/data/visualised_images/mask_pred_' + str(epoch) + '.npy', mask_pred)

            plot_segmentation_overlay(BASE, epoch, i, image, mask_pred, mask_true)

    val_loss = loss / max(num_val_batches, 1)
    dice_score = dice_score / max(num_val_batches, 1)
    iou_score = iou_score / max(num_val_batches, 1)

    net.train()
    return val_loss.item(), dice_score, iou_score


def train_model(model: nn.Module,
                device: torch.device,
                train_loader: DataLoader,
                val_loader: DataLoader,
                optimizer: torch.optim.Optimizer,
                scheduler: torch.optim.lr_scheduler.LRScheduler,
                grad_scaler: torch.cuda.amp.GradScaler,
                epochs: int,
                base_dir: str,
                checkpoint_path: Path,
                fine_tune: bool = False,
                amp: bool = False) -> tuple[list[Any], list[Any], list[float], list[float]]:
    # Initialize variables for performance visualization
    train_losses = []
    val_losses = []
    val_ious = []
    val_dices = []

    criterion = power_jaccard_loss
    # Train the model
    for epoch in range(1, epochs + 1):
        # Train for one epoch
        model.train()
        epoch_loss = 0
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                # Forward pass
                images, true_masks = batch

                images = images.to(device='cuda', dtype=torch.float32)
                true_masks = true_masks.to(device='cuda', dtype=torch.long)
                assert true_masks.min() >= 0 and true_masks.max() <= 1, 'True mask indices should be in [0, 1]'
                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    loss = criterion(masks_pred, true_masks)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                grad_scaler.step(optimizer)
                grad_scaler.update()
                # loss.backward()
                # optimizer.step()

                pbar.update(images.shape[0])
                epoch_loss += loss.item()
                pbar.set_postfix(**{'loss (batch)': loss.item()})

            if epoch % 1 == 0:
                # Evaluate on validation set
                val_loss, val_dice, val_iou = evaluate(model, val_loader, epoch=epoch, device=device, amp=amp,
                                                       BASE=base_dir,
                                                       save_fig=True)
            else:
                val_loss, val_dice, val_iou = evaluate(model, val_loader, epoch=epoch, device=device, amp=amp,
                                                       BASE=base_dir)

            scheduler.step(val_dice)

            # Update variables for performance visualization
            train_losses.append(loss.item())
            val_losses.append(val_loss)
            val_dices.append(val_dice.cpu().numpy())
            val_ious.append(val_iou.cpu().numpy())

        # Save model checkpoint
        if not fine_tune:
            Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            torch.save(state_dict, str(checkpoint_path / 'checkpoint_epoch{}.pth'.format(epoch)))
        else:
            if epoch % 1 == 0:
                Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
                state_dict = model.state_dict()
                torch.save(state_dict, str(checkpoint_path / 'fine_tune_checkpoint_epoch{}.pth'.format(epoch)))
        # Print epoch results
        # print(f'Epoch {epoch}: train_loss={loss.item():.4f}, val_loss={val_loss:.4f}, val_iou={val_iou:.4f}')

        # # Adjust learning rate
        # if epoch > 0 and epoch % 10 == 0:
        #     for g in optimizer.param_groups:
        #         g['lr'] /= 10
        pbar.update({'loss (epoch)': epoch_loss / len(train_loader)})

    return train_losses, val_losses, val_dices, val_ious


def get_transformed_dataset(manager: DataManager,
                            batch_size: int,
                            num_workers: int = 0,
                            dataset_type: str = "amateur") -> Tuple[DataLoader, DataLoader]:
    X_train, X_val, y_train, y_val = manager.load_train_val_data(dataset_type)

    # estimate mean and std for normalization
    train_mean = X_train.mean()
    train_std = X_train.std()

    train_transforms = Compose([
        NumpyToTensor(),
        ConvertImageDtype(torch.float32),
        RandomRotation(60),
        RandomAffine(degrees=0, translate=[0.2, 0.2], shear=[0.2, 0.2], scale=0.2),
        GaussianBlur(kernel_size=[3, 3], sigma=[1.0, 1.0]),
        Lambda(lambda x: torch.where(x > 0, 1, 0)),
        Normalize(train_mean, train_std),
    ])

    val_transforms = Compose([
        NumpyToTensor(),
        ConvertImageDtype(torch.float32),
        RandomRotation(60),
        RandomAffine(degrees=0, translate=[0.2, 0.2], shear=[0.2, 0.2], scale=0.2),
        GaussianBlur(kernel_size=[3, 3], sigma=[1.0, 1.0]),
        Lambda(lambda x: torch.where(x > 0, 1, 0)),
        Normalize(train_mean, train_std)
    ])

    train_dataset = MyDataset(X_train, y_train)
    augmented_dataset = MyDataset(X_train, y_train, train_transforms)
    train_dataset = ConcatDataset([train_dataset, augmented_dataset])
    val_dataset = MyDataset(X_val, y_val, val_transforms)

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              worker_init_fn=np.random.seed(42),
                              pin_memory=True)

    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            pin_memory=True)
    return train_loader, val_loader
