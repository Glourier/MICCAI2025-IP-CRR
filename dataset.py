import os
import torch
from torch.utils.data import Dataset
import pandas as pd


class MIMIC_nli(Dataset):
    """
    Load answers from NLI
    Returns a compatible Torch Dataset object customized for the MIMIC dataset.
    """

    def __init__(
            self,
            data_dir,
            data_name,
            n_concept,
            class_names=None,
            split='train',
            with_name=False,
    ):
        data_name = os.path.join(data_dir, split + data_name)
        data = pd.read_pickle(data_name)
        data['label'] = data[class_names].values.tolist()
        self.data = data.to_dict(orient='records')  # Convert DataFrame to list to speed up indexing
        self.class_names = class_names
        self.n_concept = n_concept
        self.with_name = with_name

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        study = self.data[idx]
        image_name = study['Name']
        label = torch.tensor(study['label']).long()
        columns = [f"Prediction_{i}" for i in range(self.n_concept)]
        concept = torch.tensor([study[col] for col in columns]) - 1  # [-1, 0, 1]

        if self.with_name:
            return label, concept, image_name

        return label, concept


def load_mimic_nli(data_dir, data_name='mimic_v14_hash1_pneumonia_lt.pkl', n_concept=520,
                   class_names=None,
                   with_name=False):  # From one label, get one label; no image

    trainset = MIMIC_nli(data_dir, data_name, n_concept, class_names=class_names, split='train',
                         with_name=with_name)
    valset = MIMIC_nli(data_dir, data_name, n_concept, class_names=class_names, split='val',
                       with_name=with_name)
    testset = MIMIC_nli(data_dir, data_name, n_concept, class_names=class_names, split='test',
                        with_name=with_name)

    return trainset, valset, testset
