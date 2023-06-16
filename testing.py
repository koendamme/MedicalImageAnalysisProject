from ACDCDataset import ACDCDataset
import monai
import torch
import numpy as np


def one_hot(mask):
    n_classes = 4
    one_hot = np.zeros((n_classes, mask.shape[2], mask.shape[3]))

    for i in range(n_classes):
        one_hot[i, :, :][mask[0, 0, :, :] == i] = 1

    return torch.Tensor(one_hot)[None, :, :, :]


if __name__ == '__main__':
    rootpath = "Resources/database"

    transform = monai.transforms.Compose([
        monai.transforms.AddChanneld(keys=['img', 'mask']),
        monai.transforms.ScaleIntensityd(keys=['img'], minv=0, maxv=1),
        monai.transforms.Resized(keys=['img', 'mask'], spatial_size=(-1, 200, 200))
    ])

    dataset = ACDCDataset(rootpath, "testing", pre_transform=transform, split_idxs=np.arange(10))
    dataloader = monai.data.DataLoader(dataset, batch_size=1)

    model = monai.networks.nets.UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=4,
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        num_res_units=2,
    )
    model.eval()

    dice_function = monai.metrics.DiceMetric()
    hausdorff_function = monai.metrics.HausdorffDistanceMetric()

    metrics = {
        "ES": {"Dice": [0, 0, 0, 0], "Hausdorff": [0, 0, 0, 0]},
        "ED": {"Dice": [0, 0, 0, 0], "Hausdorff": [0, 0, 0, 0]}
    }

    dice = {"ES": [[], [], [], []], "ED": [[], [], [], []]}
    hausdorff = {"ES": [[], [], [], []], "ED": [[], [], [], []]}

    for batch in dataloader:
        img = batch["img"]
        gt = batch["mask"]
        vol = batch["vol"][0]

        for i in range(img.shape[2]):
            img_slice = img[:, :, i, :, :]
            gt_slice = gt[:, :, i, :, :]

            output = model(img_slice)

            output_mask = torch.argmax(output, dim=1)[None, :, :, :]

            n_classes = 4
            for class_idx in range(n_classes):
                gt_class_mask = gt_slice == class_idx
                output_class_mask = output_mask == class_idx

                class_dice = dice_function(gt_class_mask, output_class_mask)
                class_hausdorff = hausdorff_function(gt_class_mask, output_class_mask)

                dice[vol][class_idx].append(class_dice.item())
                hausdorff[vol][class_idx].append(class_hausdorff.item())

    for i in range(4):
        for vol in ["ED", "ES"]:
            for metric in ["Dice", "Hausdorff"]:
                metrics[vol][metric][i] = sum(dice[vol][i])/len(dice[vol][i])

    print(metrics)

    # Mean DICE ED
    # Mean DICE ES
    # Mean Hausdorff ED
    # Mean Hausdorff ES
    # EF correlation
    # EF bias
    # EF standard deviation(std)
    # Volume ED correlation
    # Volume ED bias
    # Volume ED std
