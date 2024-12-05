import pandas as pd
import torch
from torch.utils.data import Dataset


class ZhangDataset(Dataset):
    def __init__(self, csv_file):
        data = pd.read_csv(csv_file)
        test_image_names = set()
        with open('data/zhanglab/test_list.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                test_image_names.add(line.strip())
        self.meta_infos = []
        for idx, d in data.iterrows():
            if d.Path not in test_image_names:
                continue
            image_path = f'data/zhanglab/test1/{d.Path.replace(".png", ".jpg")}'
            diseases = {}
            # Hardcoded for robustness and consistency
            disease_keys = ['Normal', 'Pneumonia']
            for key, value in d.iteritems():
                key = key.replace('_', ' ')
                if key in disease_keys:
                    diseases[key] = value
            disease_values = [diseases[key] for key in disease_keys]  # same order as disease_keys
            self.meta_infos.append(
                {'image_path': image_path, 'disease_keys': disease_keys, 'disease_values': disease_values})

    def __len__(self):
        return len(self.meta_infos)

    def __getitem__(self, idx):
        meta_info = self.meta_infos[idx]
        image_path = meta_info['image_path']
        keys = meta_info['disease_keys']
        labels = torch.tensor(meta_info['disease_values'], dtype=torch.float)
        return image_path, labels, keys


class CheDataset(Dataset):
    def __init__(self, csv_file):
        data = pd.read_csv(csv_file)
        test_image_names = set()
        with open('data/che/test_list.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                test_image_names.add(line.strip())
        self.meta_infos = []
        for idx, d in data.iterrows():
            if d.Path not in test_image_names:
                continue
            image_path = f'data/che/test/{d.Path.replace(".png", ".jpg")}'
            diseases = {}
            # Hardcoded for robustness and consistency
            disease_keys = ['Normal', 'Abnormal']
            for key, value in d.iteritems():
                key = key.replace('_', ' ')
                if key in disease_keys:
                    diseases[key] = value
            disease_values = [diseases[key] for key in disease_keys]  # same order as disease_keys
            self.meta_infos.append(
                {'image_path': image_path, 'disease_keys': disease_keys, 'disease_values': disease_values})

    def __len__(self):
        return len(self.meta_infos)

    def __getitem__(self, idx):
        meta_info = self.meta_infos[idx]
        image_path = meta_info['image_path']
        keys = meta_info['disease_keys']
        labels = torch.tensor(meta_info['disease_values'], dtype=torch.float)
        return image_path, labels, keys


class VinDataset(Dataset):
    def __init__(self, csv_file):
        data = pd.read_csv(csv_file)
        test_image_names = set()
        with open('data/vin/test_list.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                test_image_names.add(line.strip())
        self.meta_infos = []
        for idx, d in data.iterrows():
            if d.Path not in test_image_names:
                continue
            image_path = f'data/vin/test-clahe/{d.Path}'
            diseases = {}
            # Hardcoded for robustness and consistency
            disease_keys = ['Normal', 'Abnormal']
            for key, value in d.iteritems():
                key = key.replace('_', ' ')
                if key in disease_keys:
                    diseases[key] = value
            disease_values = [diseases[key] for key in disease_keys]  # same order as disease_keys
            self.meta_infos.append(
                {'image_path': image_path, 'disease_keys': disease_keys, 'disease_values': disease_values})

    def __len__(self):
        return len(self.meta_infos)

    def __getitem__(self, idx):
        meta_info = self.meta_infos[idx]
        image_path = meta_info['image_path']
        keys = meta_info['disease_keys']
        labels = torch.tensor(meta_info['disease_values'], dtype=torch.float)
        return image_path, labels, keys


class RSNADataset(Dataset):
    def __init__(self, csv_file):
        data = pd.read_csv(csv_file)
        test_image_names = set()
        with open('data/RSNA/test_list.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                test_image_names.add(line.strip())
        self.meta_infos = []
        for idx, d in data.iterrows():
            if d.Path not in test_image_names:
                continue
            image_path = f'data/RSNA/RSNA_images_clahe/{d.Path}'
            diseases = {}
            # Hardcoded for robustness and consistency
            disease_keys = ['Normal', 'Pneumonia']
            for key, value in d.iteritems():
                key = key.replace('_', ' ')
                if key in disease_keys:
                    diseases[key] = value
            disease_values = [diseases[key] for key in disease_keys]  # same order as disease_keys
            self.meta_infos.append(
                {'image_path': image_path, 'disease_keys': disease_keys, 'disease_values': disease_values})

    def __len__(self):
        return len(self.meta_infos)

    def __getitem__(self, idx):
        meta_info = self.meta_infos[idx]
        image_path = meta_info['image_path']
        keys = meta_info['disease_keys']
        labels = torch.tensor(meta_info['disease_values'], dtype=torch.float)
        return image_path, labels, keys


class RSNAGoodDataset(Dataset):
    def __init__(self, csv_file):
        data = pd.read_csv(csv_file)
        test_image_names = set()
        with open('data/RSNA-good/test_list.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                test_image_names.add(line.strip())
        self.meta_infos = []
        for idx, d in data.iterrows():
            if d.Path not in test_image_names:
                continue
            image_path = f'data/RSNA-good/RSNA_images_clahe/{d.Path}'
            diseases = {}
            # Hardcoded for robustness and consistency
            disease_keys = ['Normal', 'Pneumonia']
            for key, value in d.iteritems():
                key = key.replace('_', ' ')
                if key in disease_keys:
                    diseases[key] = value
            disease_values = [diseases[key] for key in disease_keys]  # same order as disease_keys
            self.meta_infos.append(
                {'image_path': image_path, 'disease_keys': disease_keys, 'disease_values': disease_values})

    def __len__(self):
        return len(self.meta_infos)

    def __getitem__(self, idx):
        meta_info = self.meta_infos[idx]
        image_path = meta_info['image_path']
        keys = meta_info['disease_keys']
        labels = torch.tensor(meta_info['disease_values'], dtype=torch.float)
        return image_path, labels, keys


class COVIDDataset(Dataset):
    def __init__(self, csv_file):
        data = pd.read_csv(csv_file)
        test_image_names = set()
        with open('/home/image023/data/Xplainer-master/data/COVID/test_list1.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                test_image_names.add(line.strip())
        self.meta_infos = []
        for idx, d in data.iterrows():
            if d.Path not in test_image_names:
                continue
            image_path = f'/home/image023/data/Xplainer-master/data/COVID/images_merge/{d.Path}'
            diseases = {}
            # Hardcoded for robustness and consistency
            disease_keys = ['No Finding', 'COVID-19']
            for key, value in d.iteritems():
                key = key.replace('_', ' ')
                if key in disease_keys:
                    diseases[key] = value
            disease_values = [diseases[key] for key in disease_keys]  # same order as disease_keys
            self.meta_infos.append(
                {'image_path': image_path, 'disease_keys': disease_keys, 'disease_values': disease_values})

    def __len__(self):
        return len(self.meta_infos)

    def __getitem__(self, idx):
        meta_info = self.meta_infos[idx]
        image_path = meta_info['image_path']
        keys = meta_info['disease_keys']
        labels = torch.tensor(meta_info['disease_values'], dtype=torch.float)
        return image_path, labels, keys


class COVIDXDataset(Dataset):
    def __init__(self, csv_file):
        data = pd.read_csv(csv_file)
        test_image_names = set()
        with open('/home/image023/data/Xplainer-master/data/COVIDX/test_list1.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                test_image_names.add(line.strip())
        self.meta_infos = []
        for idx, d in data.iterrows():
            if d.Path not in test_image_names:
                continue
            image_path = f'/home/image023/data/Xplainer-master/data/COVIDX/images_merge/{d.Path}'
            diseases = {}
            # Hardcoded for robustness and consistency
            disease_keys = ['No Finding', 'COVID-19']
            for key, value in d.iteritems():
                key = key.replace('_', ' ')
                if key in disease_keys:
                    diseases[key] = value
            disease_values = [diseases[key] for key in disease_keys]  # same order as disease_keys
            self.meta_infos.append(
                {'image_path': image_path, 'disease_keys': disease_keys, 'disease_values': disease_values})

    def __len__(self):
        return len(self.meta_infos)

    def __getitem__(self, idx):
        meta_info = self.meta_infos[idx]
        image_path = meta_info['image_path']
        keys = meta_info['disease_keys']
        labels = torch.tensor(meta_info['disease_values'], dtype=torch.float)
        return image_path, labels, keys


class COVID19Dataset(Dataset):
    def __init__(self, csv_file):
        data = pd.read_csv(csv_file)
        test_image_names = set()
        with open('/home/image023/data/Xplainer-master/data/COVID19/test_list2.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                test_image_names.add(line.strip())
        self.meta_infos = []
        for idx, d in data.iterrows():
            if d.Path not in test_image_names:
                continue
            image_path = f'/home/image023/data/Xplainer-master/data/COVID19/images_merge/{d.Path}'
            diseases = {}
            # Hardcoded for robustness and consistency
            disease_keys = ['No Finding', 'COVID-19']
            for key, value in d.iteritems():
                key = key.replace('_', ' ')
                if key in disease_keys:
                    diseases[key] = value
            disease_values = [diseases[key] for key in disease_keys]  # same order as disease_keys
            self.meta_infos.append(
                {'image_path': image_path, 'disease_keys': disease_keys, 'disease_values': disease_values})

    def __len__(self):
        return len(self.meta_infos)

    def __getitem__(self, idx):
        meta_info = self.meta_infos[idx]
        image_path = meta_info['image_path']
        keys = meta_info['disease_keys']
        labels = torch.tensor(meta_info['disease_values'], dtype=torch.float)
        return image_path, labels, keys


class MultiDataset(Dataset):
    def __init__(self, csv_file):
        data = pd.read_csv(csv_file)
        test_image_names = set()
        with open('data/MulCenter_test/test/dataset9/test_list.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                test_image_names.add(line.strip())
        self.meta_infos = []
        for idx, d in data.iterrows():
            if d.Path not in test_image_names:
                continue
            image_path = f'data/MulCenter_test/test/dataset9/images-clahe/{d.Path}'
            diseases = {}
            # Hardcoded for robustness and consistency
            disease_keys = ['Normal', 'Pneumonia']
            for key, value in d.iteritems():
                key = key.replace('_', ' ')
                if key in disease_keys:
                    diseases[key] = value
            disease_values = [diseases[key] for key in disease_keys]  # same order as disease_keys
            self.meta_infos.append(
                {'image_path': image_path, 'disease_keys': disease_keys, 'disease_values': disease_values})

    def __len__(self):
        return len(self.meta_infos)

    def __getitem__(self, idx):
        meta_info = self.meta_infos[idx]
        image_path = meta_info['image_path']
        keys = meta_info['disease_keys']
        labels = torch.tensor(meta_info['disease_values'], dtype=torch.float)
        return image_path, labels, keys