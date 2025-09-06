from torch.utils.data import DataLoader, Dataset
import torch
from torch import nn as nn


class NewsDataset(Dataset):
    """新闻推荐数据集"""
    def __init__(self, features, labels, user_history, max_seq_len=50):
        self.features = features
        self.labels = labels
        self.user_history = user_history
        self.max_seq_len = max_seq_len
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        # 获取特征
        feature = self.features[idx]
        label = self.labels[idx] if self.labels is not None else 0
        
        # 获取用户历史序列
        user_id = int(feature['user_id'])
        article_id = int(feature['article_id'])
        
        # 用户历史点击序列
        if user_id in self.user_history:
            hist_items = self.user_history[user_id]['items'][-self.max_seq_len:]
            hist_cats = self.user_history[user_id]['categories'][-self.max_seq_len:]
            hist_times = self.user_history[user_id]['timestamps'][-self.max_seq_len:]
        else:
            hist_items = []
            hist_cats = []
            hist_times = []
        
        # 序列长度
        seq_len = len(hist_items)
        
        # Padding
        if seq_len < self.max_seq_len:
            pad_len = self.max_seq_len - seq_len
            hist_items = [0] * pad_len + hist_items
            hist_cats = [0] * pad_len + hist_cats
            hist_times = [0] * pad_len + hist_times
        """
        user_id:当前用户编码后的id
        article_id: 推荐文章待编码后的id
        category_id： 目标文章所属类别的编码后ID
        hist_items: 用户历史点击的文章的ID序列
        hist_cats: 历史点击文章对应的类别ID序列
        hist_times: 历史点击的时间戳序列
        seq_len: 用户实际历史行为的长度
        features: 标准化后的数值特征向量
        label：目标标签，表示用户是否被点击
        """
        return {
            'user_id': user_id,
            'article_id': article_id,
            'category_id': int(feature['category_id']),
            'hist_items': torch.LongTensor(hist_items),
            'hist_cats': torch.LongTensor(hist_cats),
            'hist_times': torch.FloatTensor(hist_times),
            'seq_len': seq_len,
            'features': torch.FloatTensor(feature['numeric_features']),
            'label': torch.FloatTensor([label])
        }