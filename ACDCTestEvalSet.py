import numpy as np
import monai
import glob
import os
import SimpleITK as sitk


class ACDCTestEvalSet(monai.data.Dataset):
    def __init__(self,
                 rootpath,
                 pre_transform=None):
        self.path = rootpath
        self.pre_transform = pre_transform
        self.data = []
        self.load_data()
        if self.pre_transform:
            self._perform_pre_transform()

    def load_data(self):
        """
        returns dict{2dimg, 2dmask}
        """

        directories = next(os.walk(self.path))[1]

        for patient in directories:
            patient_paths = glob.glob(os.path.join(self.path, patient, "*.gz"))

            self.load_patient(patient_paths)

    def load_patient(self, patient_paths):
        image = sitk.ReadImage(patient_paths[0])
        image_array = sitk.GetArrayFromImage(image)
        self.data.append({"img": image_array[0], "mask": np.zeros(image_array.shape), "path": patient_paths[0]})

    def _perform_pre_transform(self):
        if not self.pre_transform:
            raise Exception(
                "_perform_transforms should only be called when transform is not None"
            )

        for i in range(len(self.data)):
            self.data[i] = self.pre_transform(self.data[i])

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
