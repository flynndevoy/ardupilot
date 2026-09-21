import time
from contextlib import contextmanager
import numpy as np
import warnings
import math
import yaml
import skimage.transform
from typing import Sized, Optional

def argmax2d(x):
    return np.unravel_index(x.argmax(), x.shape)


def argmin2d(x):
    return np.unravel_index(x.argmin(), x.shape)


def build_geom_transform(
        dst_w,
        dst_h,
        src_center_x,
        src_center_y,
        scale_x=1.0,
        scale_y=1.0,
        angle=0.0,
        shear=0.0,
        hflip=False,
        vflip=False,
        return_params=False
):
    """
    @param dst_size_x:
    @param dst_size_y:
    @param src_center_x:
    @param src_center_y:
    @param scale_x:
    @param scale_y:
    @param angle:
    @param shear:
    @param hflip:
    @param vflip:
    @return: transform matrix

    Usage with cv2:

    crop = cv2.warpAffine(img, np.linalg.pinv(tform.params)[:2, :],
                          dsize=(self.frame_size, self.frame_size), flags=cv2.INTER_LINEAR)
    """

    if hflip:
        scale_x *= -1
    if vflip:
        scale_y *= -1

    tform = skimage.transform.AffineTransform(translation=(src_center_x, src_center_y))
    tform = skimage.transform.AffineTransform(scale=(1.0 / scale_x, 1.0 / scale_y)) + tform
    tform = skimage.transform.AffineTransform(rotation=angle * math.pi / 180,
                                              shear=shear * math.pi / 180) + tform
    tform = skimage.transform.AffineTransform(translation=(-dst_w / 2, -dst_h / 2)) + tform

    if return_params:
        return np.linalg.pinv(tform.params)
    else:
        return tform



def normalize_experiment_name(experiment_name: str):
    if experiment_name.startswith("experiments/"):
        experiment_name = experiment_name[len("experiments/") :]
    if experiment_name.endswith(".yaml"):
        experiment_name = experiment_name[: -len(".yaml")]

    return experiment_name


def load_config_data(experiment_name: str, experiments_dir='experiments') -> dict:
    with open(f"{experiments_dir}/{experiment_name}.yaml") as f:
        cfg: dict = yaml.load(f, Loader=yaml.FullLoader)
    return cfg


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class DotDict(dict):
    """dot.notation access to dictionary attributes

    Refer: https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary/23689767#23689767
    """  # NOQA

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


@contextmanager
def timeit_context(name):
    startTime = time.time()
    yield
    elapsedTime = time.time() - startTime
    print(f"[{name}] finished in {elapsedTime:0.3f}s")


def print_stats(title, array):
    if len(array):
        print(
            "{} shape:{} dtype:{} min:{} max:{} mean:{} median:{}".format(
                title,
                array.shape,
                array.dtype,
                np.min(array),
                np.max(array),
                np.mean(array),
                np.median(array),
            )
        )
    else:
        print(title, "empty")