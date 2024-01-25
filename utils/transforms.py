import cv2
import numpy as np
from torchvision.transforms.v2 import Transform
from torchvision import tv_tensors
from torchvision.tv_tensors import Mask


class RemoveSmallVessel(Transform):
    """
    Remove small vessels from a binary mask.

    It works as a torchvision.transforms.v2 transforms,
    so it will only be used on torchvision.tv_tensors.Mask.

    Make sure to use a mask as input or to use `with tv_tensors.set_return_type("TVTensor"):` before calling this transform.
    For now, it only works when a single image or a tuple / list of images is passed as input.

    Parameters
    ----------

    min_area : int
        Minimum area of the object to be kept.

    """

    def __init__(self, min_area=50):
        super().__init__()
        self.min_area = min_area

    def __call__(self, *images):
        with tv_tensors.set_return_type("TVTensor"):
            transformed_images = []
            for img in images:
                if type(img) == Mask:
                    img = self._remove_small_objects(img, self.min_area)
                if len(images) == 1:
                    return img
                transformed_images.append(img)
        return tuple(transformed_images)

    def _remove_small_objects(self, mask, min_area):
        mask = mask.numpy()
        if mask.ndim == 2:
            return Mask(remove_small_objects(mask, min_area))
        elif mask.ndim != 3:
            raise ValueError("Mask must be 2D or 3D, got {}D".format(mask.ndim))
        else:
            num_images = mask.shape[0]
            new_mask = np.zeros_like(mask)
            for i in range(num_images):
                new_mask[i] = remove_small_objects(mask[i], min_area)
            return Mask(new_mask)


def remove_small_objects(img, min_area):
    """Remove small objects from a binary image.

    Parameters
    ----------
    img : np.ndarray
    min_area : int
        Minimum area of the object to be kept.

    Returns
    -------
    np.ndarray
        Image with small objects removed.
    """
    img = img.astype(np.uint8)
    contours = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    for c in contours:
        if cv2.contourArea(c) < min_area:
            cv2.drawContours(img, [c], -1, 0, -1)
    return img


if __name__ == "__main__":
    import torch

    x = torch.zeros((2, 100, 100))
    x[..., :2, :2] = 1  # Area is 4
    x[..., 5:10, 5:10] = 1  # Area is 25
    x[..., 50:60, 50:60] = 1  # Area is 100
    x[..., 80:100, 80:100] = 1  # Area is 400
    x = Mask(x)

    transform = RemoveSmallVessel(min_area=50)

    with tv_tensors.set_return_type("TVTensor"):
        x = transform(x)

    assert type(x) == Mask, "Return type is not Mask, got {}".format(type(x))
    assert x.shape == (2, 100, 100), "Shape is not the same, got {}".format(x.shape)
    assert x.sum() == 2 * 500, "Sum is not the same, got {}".format(x.sum())
