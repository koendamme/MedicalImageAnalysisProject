import numpy as np
from tqdm import tqdm
import monai
import torch
from ACDCDataset import ACDCDataset
import os
from custom_transforms import GroundTruthTransform
import json
from datetime import datetime
import wandb
from sklearn.model_selection import KFold
from eval_metrics import compute_metric



def log_data(data):
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    with open(f"training_log/{now}.json", 'w') as f:
        json.dump(data, f)


def wandb_masks(mask_output, mask_gt):
    """ Function that generates a mask dictionary in format that W&B requires """
    sigmoid = torch.nn.Sigmoid()
    mask_output = sigmoid(mask_output)

    mask_output = torch.argmax(mask_output, dim=0)
    print(torch.max(mask_output))

    mask_gt = torch.argmax(mask_gt, dim=0)

    # Transform masks to numpy arrays on CPU
    # Note: .squeeze() removes all dimensions with a size of 1 (here, it makes the tensors 2-dimensional)
    # Note: .detach() removes a tensor from the computational graph to prevent gradient computation for it
    mask_output = mask_output.squeeze().detach().cpu().numpy()
    mask_gt = mask_gt.squeeze().detach().cpu().numpy()

    # Create mask dictionary with class label and insert masks
    class_labels = {0: 'Background', 1: 'Class 1', 2: 'Class 2', 3: 'Class 3'}
    masks = {
        'predictions': {'mask_data': mask_output, 'class_labels': class_labels},
        'ground truth': {'mask_data': mask_gt, 'class_labels': class_labels}
    }
    return masks


def log_to_wandb(epoch, train_loss, val_loss, batch_data, outputs):
    """ Function that logs ongoing training variables to W&B """

    # Create list of images that have segmentation masks for model output and ground truth
    log_imgs = []
    for img, mask_output, mask_gt in zip(batch_data['img'], outputs, batch_data['mask']):
        masks = wandb_masks(mask_output, mask_gt)
        log_img = wandb.Image(img, masks=masks)

        log_imgs.append(log_img)

    # Send epoch, losses and images to W&B
    wandb.log({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss, 'results': log_imgs})


def train_model(dataset_path, device, loss_function, lr, num_epochs, transforms, channels):
    kfold = KFold(n_splits=5, shuffle=True)
    trained_models = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(np.arange(100))):
        run = wandb.init(
            project='ACDC Project',
            name=f'Test run at {datetime.now().strftime("%Y%m%d-%H%M%S")}',
            config={
                'loss function': str(loss_function),
                'lr': lr,
                'batch_size': 16,
            }
        )

        run_id = run.id

        model = monai.networks.nets.UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=4,
            channels=channels,
            strides=(2, 2, 2),
            num_res_units=2,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        train_dataset = ACDCDataset(dataset_path, "training", train_idx, transform=transforms)
        val_dataset = ACDCDataset(dataset_path, "training", val_idx, transform=transforms)

        train_loader = monai.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = monai.data.DataLoader(val_dataset, batch_size=16, shuffle=True)

        for epoch in range(num_epochs):
            print(f"Epoch: {epoch+1}/{num_epochs}")
            model.train()
            train_loss = 0
            step = 0
            print("Training...")
            for batch in tqdm(train_loader):
                step += 1
                x_batch = batch['img'].to(device)
                y_batch = batch['mask'].to(device)

                outputs = model(x_batch)
                loss = loss_function(outputs, y_batch)

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss = train_loss/step

            step = 0
            model.eval()
            val_loss = 0
            print("Validating...")
            for batch in tqdm(val_loader):
                x_batch = batch['img'].to(device)
                y_batch = batch['mask'].to(device)
                step += 1
                outputs = model(x_batch)
                loss = loss_function(outputs, y_batch)
                val_loss += loss.item()

            val_loss = val_loss/step
            print(f"{epoch}: Training/Validation loss: {train_loss:.4f}/{val_loss:.4f}")
            log_to_wandb(epoch, train_loss, val_loss, batch, outputs)

        trained_models.append(model)
        run.finish()

    return trained_models


if __name__ == '__main__':
    data_path = "Resources/database"

    if not os.path.exists(data_path):
        print("Please update your data path to an existing folder.")
    elif not {"training", "testing"}.issubset(set(os.listdir(data_path))):
        print("Please update your data path to the correct folder (should contain train, val and test folders).")
    else:
        print("Congrats! You selected the correct folder :)")

    transforms = monai.transforms.Compose([
        monai.transforms.AddChanneld(keys=['img', 'mask']),
        # monai.transforms.NormalizeIntensityd(keys='img', subtrahend=67.27, divisor=84.66),
        monai.transforms.Resized(keys=['img', 'mask'], spatial_size=(200, 200)),
        GroundTruthTransform()
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'The used device is {device}')

    dice = monai.losses.DiceLoss(sigmoid=True, batch=True)
    dice_ce = monai.losses.DiceCELoss(sigmoid=True, batch=True)
    dice_focal = monai.losses.DiceFocalLoss(sigmoid=True, batch=True)

    wandb.login(key="02febfddb1e6757681c2f1e1257c49e0b3d57bc9")

    for loss_function in [dice, dice_ce, dice_focal]:
        _ = train_model(
            dataset_path=data_path,
            device=device,
            loss_function=loss_function,
            lr=1e-3,
            num_epochs=20,
            transforms=transforms,
            channels=(16, 32, 64, 128))

    for lr in [5e-4, 8e-4, 1e-3, 12e-4, 15e-4]:
        _ = train_model(
            dataset_path=data_path,
            device=device,
            loss_function=dice_focal,
            lr=lr,
            num_epochs=20,
            transforms=transforms,
            channels=(16, 32, 64, 128))

    channels = [
        (32, 64, 128, 256),
        (16, 32, 64, 128),
        (32, 64, 128, 256),
        (16, 32, 64, 128),
        (32, 64, 128, 256),
        (16, 32, 64, 128)
    ]

    for channel in channels:
        _ = train_model(
            dataset_path=data_path,
            device=device,
            loss_function=dice_focal,
            lr=lr,
            num_epochs=20,
            transforms=transforms,
            channels=channel)
