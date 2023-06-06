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
from custom_loss import CoshDiceLoss
import math
import matplotlib.pyplot as plt
from PIL import Image


def log_data(data):
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    with open(f"training_log/{now}.json", 'w') as f:
        json.dump(data, f)


def wandb_masks(mask_output, mask_gt):
    """ Function that generates a mask dictionary in format that W&B requires """
    sigmoid = torch.nn.Sigmoid()
    mask_output = sigmoid(mask_output)
    class0 = mask_output[0, :, :]
    class1 = mask_output[1, :, :]
    class2 = mask_output[2, :, :]
    class3 = mask_output[3, :, :]

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


def train_model(model, train_loader, val_loader, device, loss_function, optimizer, num_epochs):
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

    return model


if __name__ == '__main__':
    data_path = "Resources"

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

    train_dataset = ACDCDataset(data_path, "training", transforms)
    test_dataset = ACDCDataset(data_path, "testing", transforms)

    train_loader = monai.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = monai.data.DataLoader(test_dataset, batch_size=16, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'The used device is {device}')

    model = monai.networks.nets.UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=4,
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        num_res_units=2,
    ).to(device)

    # loss_function = monai.losses.DiceLoss(sigmoid=True, batch=True, include_background=False)
    # loss_function = monai.losses.DiceCELoss(sigmoid=True, batch=True)
    loss_function = monai.losses.DiceFocalLoss(sigmoid=True, batch=True, include_background=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    wandb.login()
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    run = wandb.init(
        project='ACDC Project',
        name=f'Test run at {now}',
        config={
            'loss function': str(loss_function),
            'lr': optimizer.param_groups[0]["lr"],
            'batch_size': train_loader.batch_size,
        }
    )

    run_id = run.id

    trained = train_model(model, train_loader, test_loader, device, loss_function, optimizer, 30)

    torch.save(trained.state_dict(), r'trainedUNet3.pt')
    run.finish()






