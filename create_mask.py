import os
import glob
from ACDCTestEvalSet import ACDCTestEvalSet
import monai
import torch
import numpy as np
import nibabel as nib


if __name__ == '__main__':
    transform = monai.transforms.Compose([
        monai.transforms.AddChanneld(keys=['img', 'mask']),
        monai.transforms.ScaleIntensityd(keys=['img'], minv=0, maxv=1),
        # monai.transforms.Resized(keys=['img'], spatial_size=(-1, 200, 200))
    ])

    dataset = ACDCTestEvalSet("Resources/secret_test", transform)

    dataloader = monai.data.DataLoader(dataset, batch_size=1)

    model = monai.networks.nets.UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=4,
        channels=(32, 64, 128, 256),
        strides=(2, 2, 2),
        num_res_units=2,
    )
    model.load_state_dict(torch.load("final_models/epoch55.pt"))
    model.eval()
    inferer = monai.inferers.SlidingWindowInferer(roi_size=[200, 200])

    for batch in dataloader:
        img = batch["img"]
        new_mask = np.zeros(img.shape)
        path = batch["path"][0]
        img_name = path.split("/")[-1]

        for i in range(img.shape[2]):
            img_slice = img[:, :, i, :, :]
            output = inferer(img_slice, network=model)
            # output = model(img_slice)
            output_mask = torch.argmax(output, dim=1)[None, :, :, :]
            new_mask[0, 0, i, :, :] = output_mask

        file_path = os.path.join("Resources", "secret_test_output", img_name.split("\\")[-1].split(".")[0] + "_seg.nii.gz")
        # file_name = "Resources/secret_test_output/" + img_name.split(".")[0] + "_seg.nii.gz"
        # squeezed = new_mask.squeeze()
        file = nib.Nifti1Image(new_mask.squeeze().T, np.eye(4))
        nib.save(file, file_path)

