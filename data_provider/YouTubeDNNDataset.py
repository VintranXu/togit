import torch
from torch.utils.data import Dataset

class YouTubeDNNDataset(Dataset):
    """YouTube DNN数据集"""
    def __init__(self, user_ids, histories, target_items, labels):
        self.user_ids = torch.LongTensor(user_ids)
        self.histories = torch.LongTensor(histories)
        self.target_items = torch.LongTensor(target_items)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.user_ids)
    
    def __getitem__(self, idx):
        return {
            'user_id': self.user_ids[idx],
            'history': self.histories[idx],
            'target_item': self.target_items[idx],
            'label': self.labels[idx]
        }