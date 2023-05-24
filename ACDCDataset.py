import numpy as np
import monai
import glob
import os
import SimpleITK as sitk


class ACDCDataset(monai.data.Dataset):
    def __init__(self, rootpath, mode, transform=None):
        if mode not in ["training", "testing"]:
            raise Exception("must be either training or testing for the dataset to be loaded")

        self.path = os.path.join(rootpath, mode)
        self.transform = transform
        self.data = []
        self.load_data()

    def load_data(self):
        """
        returns dict{2dimg, 2dmask}
        """
        for patient in next(os.walk(self.path))[1]:
            patient_paths = glob.glob(os.path.join(self.path, patient, '*.gz'))

            patient_paths.sort()
            self.load_patient(patient_paths)

    def load_patient(self, patient_paths):
        for combi in [(1, 2), (3, 4)]:
            image = sitk.ReadImage(patient_paths[combi[0]])
            image_array = sitk.GetArrayFromImage(image)

            mask = sitk.ReadImage(patient_paths[combi[1]])
            mask_array = sitk.GetArrayFromImage(mask)

            for i in range(image_array.shape[0]):
                dictionary = {}
                dictionary['img'] = image_array[i, :, :]
                dictionary['mask'] = mask_array[i, :, :]

                self.data.append(dictionary)

    def __getitem__(self, index):
        # Make getitem return a dictionary with keys ['img', 'label'] for the image and label respectively
        item = self.data[index]
        if self.transform:
            item = self.transform(item)
        return item

    def get_total_meansd(self):
        norm = []
        for x in self.data:
          norm.append(x["img"])

        norm = np.array(norm)
        return np.mean(norm), np.std(norm)

    def __len__(self):
        return len(self.data)