import numpy as np


class GroundTruthTransform(object):
    """
    Transform the ground truth so that multi class segmentation is possible.
    """

    def __call__(self, sample):
        mask = sample["mask"]
        n_classes = 4

        one_hot = np.zeros((n_classes, mask.shape[1], mask.shape[2]))

        for i in range(n_classes):
            one_hot[i, :, :][mask[0, :, :] == i] = 1

        return {"img": sample["img"], "mask": one_hot}
