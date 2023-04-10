import collections
import random
import torch
from torch.utils.data import Dataset
# import from torchvision segmentation ConvertImageDtype, RandomCrop, RandomHorizontalFlip, RandomVerticalFlip, Resize, Compose

class EnhancedCompose(object):
    """Composes several transforms together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            if isinstance(t, collections.Sequence):
                assert isinstance(img, collections.Sequence) and len(img) == len(
                    t), "size of image group and transform group does not fit"
                tmp_ = []
                for i, im_ in enumerate(img):
                    if callable(t[i]):
                        tmp_.append(t[i](im_))
                    else:
                        tmp_.append(im_)
                img = tmp_
            elif callable(t):
                img = t(img)
            elif t is None:
                continue
            else:
                raise Exception('unexpected type')
        return img


# class MyDataset(Dataset):
#     def __init__(self, X, y, transform=None, mask_transform=None):
#         self.X = X
#         self.y = y
#         self.transform = transform
#         self.mask_transform = mask_transform
#         self.seed = 42
#
#     def __len__(self):
#         return len(self.X)
#
#     def __getitem__(self, idx):
#         x = torch.from_numpy(self.X[idx])
#         y = torch.from_numpy(self.y[idx]).float()
#
#         if self.transform:
#             random.seed(self.seed)
#             torch.manual_seed(self.seed)
#             x, y = self.transform([x, y])
#
#         return x, y.long()


class MyDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform
        self.seed = 42

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x, y = self.X[idx], self.y[idx]

        if self.transform:
            x, y = self.transform(x, y)
        else:
            x, y = torch.from_numpy(x), torch.from_numpy(y).long()

        return x, y
