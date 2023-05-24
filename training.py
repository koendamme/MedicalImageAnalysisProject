from tqdm import tqdm
import numpy as np
import monai
import torch
from ACDCDataset import ACDCDataset
import os


data_path = "database"

if not os.path.exists(data_path):
    print("Please update your data path to an existing folder.")
elif not set(["training", "testing"]).issubset(set(os.listdir(data_path))):
    print("Please update your data path to the correct folder (should contain train, val and test folders).")
else:
    print("Congrats! You selected the correct folder :)")

transforms = monai.transforms.Compose([
    monai.transforms.AddChanneld(keys=['img', 'mask']),
    monai.transforms.NormalizeIntensityd(keys='img', subtrahend=67.27, divisor=84.66),
    monai.transforms.Resized(keys=['img', 'mask'], spatial_size=(200, 200))
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
    out_channels=1,
    channels=(16, 32, 64, 128),
    strides=(2, 2, 2, 2),
    num_res_units=2,
).to(device)

loss_function = monai.losses.DiceLoss(sigmoid=True, batch=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 40

dataloaders = {'train': train_loader, 'val': test_loader}

for epoch in range(num_epochs):
    print(f"Epoch: {epoch+1}/{num_epochs}")
    epoch_losses = {'train': [], 'val': []}
    batch_data = {'train': [], 'val': []}
    outputs = {'train': [], 'val': []}

    for mode in ['train', 'val']:
    # for mode in ['train']:
        print(f"Current mode: {mode}")
        for i, batch in enumerate(tqdm(dataloaders[mode])):
            # batch_data[mode].extend(batch['img'])
            # batch_data[mode].extend(batch['mask'])
            x_batch = batch['img'].to(device)
            y_batch = batch['mask'].to(device)

            output = model(x_batch)

            loss = loss_function(output, y_batch)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # outputs[mode].extend(output.cpu().numpy)
            epoch_losses[mode].append(loss.item())
        print(f"Mean loss in {mode} mode: {np.mean(epoch_losses[mode])}")

    # log_to_wandb(epoch, epoch_loss['train'], epoch_loss['val'], batch_data['train'], outputs['train'])


# Store the network parameters
torch.save(model.state_dict(), r'trainedUNet.pt')