import numpy as np
import monai
import glob
import os
import SimpleITK as sitk


class ACDCDataset(monai.data.Dataset):
    def __init__(self, rootpath, mode, split_idxs, transform=None):
        if mode not in ["training", "testing"]:
            raise Exception(
                "must be either training, testing or cross for the dataset to be loaded"
            )

        self.path = os.path.join(rootpath, mode)
        self.split_idxs = split_idxs
        self.transform = transform
        self.data = []
        self.load_data()
        if self.transform:
            self._perform_transforms()

    def load_data(self):
        """
        returns dict{2dimg, 2dmask}
        """

        for patient in np.array(next(os.walk(self.path))[1])[self.split_idxs]:
            patient_paths = glob.glob(os.path.join(self.path, patient, "*.gz"))

            patient_paths.sort()
            self.load_patient(patient_paths)

    def load_patient(self, patient_paths):
        for combi in [(1, 2), (3, 4)]:
            image = sitk.ReadImage(patient_paths[combi[0]])
            image_array = sitk.GetArrayFromImage(image)

            mask = sitk.ReadImage(patient_paths[combi[1]])
            mask_array = sitk.GetArrayFromImage(mask)

            for i in range(image_array.shape[0]):
                dictionary = {"img": image_array[i, :, :], "mask": mask_array[i, :, :]}

                self.data.append(dictionary)

    def _perform_transforms(self):
        if not self.transform:
            raise Exception(
                "_perform_transforms should only be called when transform is not None"
            )

        for i in range(len(self.data)):
            self.data[i] = self.transform(self.data[i])

    def __getitem__(self, index):
        # Make getitem return a dictionary with keys ['img', 'label'] for the image and label respectively
        item = self.data[index]
        return item

    def get_total_meansd(self):
        norm = []
        for x in self.data:
            norm.append(x["img"])

        norm = np.array(norm)
        return np.mean(norm), np.std(norm)

    def __len__(self):
        return len(self.data)
