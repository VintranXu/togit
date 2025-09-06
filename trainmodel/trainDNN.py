import argparse
import os
import pickle
import random
import sys
import warnings

import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from tqdm import tqdm

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.YouTubeDNNModel import YouTubeDNN
from data_provider.YouTubeDNNDataset import YouTubeDNNDataset
from utils import Logger

warnings.filterwarnings('ignore')

# 设置随机种子
seed = 2020
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

# 命令行参数
parser = argparse.ArgumentParser(description='YouTube DNN模型训练')
parser.add_argument('--mode', default='valid', choices=['valid', 'online'],
                    help='训练模式：valid(离线验证) 或 online(在线)')
parser.add_argument('--logfile', default='train_youtube_dnn.log', help='日志文件名')
parser.add_argument('--embedding_dim', default=64, type=int, help='Embedding维度')
parser.add_argument('--hidden_units', default=[256, 128], nargs='+', type=int, 
                    help='隐藏层单元数')
parser.add_argument('--batch_size', default=16384, type=int, help='批大小')
parser.add_argument('--epochs', default=50, type=int, help='训练轮数')
parser.add_argument('--learning_rate', default=0.01, type=float, help='学习率')
parser.add_argument('--test_ratio', default=0.2, type=float, help='验证集比例')
parser.add_argument('--max_seq_len', default=50, type=int, help='最大序列长度')
parser.add_argument('--patience', default=2, type=int, help='早停轮数')
parser.add_argument('--num_workers', default=4, type=int, help='数据加载进程数')

args = parser.parse_args()

# 设备选择
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 初始化日志
os.makedirs('/home/wangtiantian/shikainan/newscommemder/top2wk/user_data/log', exist_ok=True)
log = Logger(f'/home/wangtiantian/shikainan/newscommemder/top2wk/user_data/log/{args.logfile}').logger
log.info(f'YouTube DNN模型训练开始，mode: {args.mode}, device: {device}')
log.info(f'训练参数: {vars(args)}')


def prepare_training_data(df_click, test_ratio=0.2, max_seq_len=50):
    """准备训练数据"""
    log.info("开始准备训练数据...")
    
    # 编码用户和物品ID
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    
    # 编码后的ID需要+1，让0作为padding专用
    df_click['user_id_encoded'] = user_encoder.fit_transform(df_click['user_id']) + 1
    df_click['item_id_encoded'] = item_encoder.fit_transform(df_click['click_article_id']) + 1
    
    log.info(f"用户数量: {len(user_encoder.classes_)}")
    log.info(f"物品数量: {len(item_encoder.classes_)}")
    
    # 构建训练样本
    train_samples = []
    
    grouped = df_click.groupby('user_id_encoded')
    for user_id, group in tqdm(grouped, desc="构建训练样本"):
        items = group['item_id_encoded'].values
        timestamps = group['click_timestamp'].values
        
        # 按时间排序
        sorted_indices = np.argsort(timestamps)
        items = items[sorted_indices]
        
        # 为每个位置的物品构建历史序列和负样本
        start_idx = max(1, len(items) - 3)  # 只采样最后3个位置
        for i in range(start_idx, len(items)):
            hist = items[:i]  # 历史序列
            target = items[i]  # 目标物品
            
            # 正样本
            train_samples.append({
                'user_id': user_id,
                'history': hist,
                'target_item': target,
                'label': 1
            })
            
            # 负样本：随机选择用户未交互过的物品
            neg_item = random.choice(range(1, len(item_encoder.classes_) + 1))
            while neg_item in items:  # 确保是负样本
                neg_item = random.choice(range(1, len(item_encoder.classes_) + 1))
            
            train_samples.append({
                'user_id': user_id,
                'history': hist,
                'target_item': neg_item,
                'label': 0
            })
    
    log.info(f"生成训练样本数: {len(train_samples)}")
    
    # 打乱训练数据
    random.shuffle(train_samples)
    
    # 处理变长序列，padding到固定长度
    user_ids = []
    histories = []
    target_items = []
    labels = []
    
    for sample in train_samples:
        user_ids.append(sample['user_id'])
        
        # padding历史序列
        hist = sample['history']
        if len(hist) > max_seq_len:
            hist = hist[-max_seq_len:]  # 取最近的max_seq_len个
        else:
            # 进行前填充
            hist = np.pad(hist, (max_seq_len - len(hist), 0), 
                         mode='constant', constant_values=0)
        
        histories.append(hist)
        target_items.append(sample['target_item'])
        labels.append(sample['label'])
    
    user_ids = np.array(user_ids)
    histories = np.array(histories)
    target_items = np.array(target_items)
    labels = np.array(labels)
    
    # 划分训练和验证集
    n_train = int(len(user_ids) * (1 - test_ratio))
    
    train_data = (user_ids[:n_train], histories[:n_train], 
                  target_items[:n_train], labels[:n_train])
    val_data = (user_ids[n_train:], histories[n_train:], 
                target_items[n_train:], labels[n_train:])
    
    log.info(f"训练样本数: {len(train_data[0])}")
    log.info(f"验证样本数: {len(val_data[0])}")
    
    return train_data, val_data, user_encoder, item_encoder


def train_model(model, train_loader, val_loader, epochs, learning_rate, patience=3):
    """训练模型"""
    log.info("开始训练模型...")
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    model.to(device)
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None  # 保存最佳模型状态
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        for batch in train_bar:
            user_ids = batch['user_id'].to(device)
            histories = batch['history'].to(device)
            target_items = batch['target_item'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            scores, _, _ = model(user_ids, histories, target_items)
            loss = criterion(scores, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predictions = torch.sigmoid(scores) > 0.5
            train_correct += (predictions == labels.bool()).sum().item()
            train_total += labels.size(0)
            
            train_bar.set_postfix({
                'loss': loss.item(),
                'acc': train_correct / train_total
            })

        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total
        
        if epoch % 10 == 0:
            # 验证阶段
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]')
                for batch in val_bar:
                    user_ids = batch['user_id'].to(device)
                    histories = batch['history'].to(device)
                    target_items = batch['target_item'].to(device)
                    labels = batch['label'].to(device)
                    
                    scores, _, _ = model(user_ids, histories, target_items)
                    loss = criterion(scores, labels)
                    
                    val_loss += loss.item()
                    predictions = torch.sigmoid(scores) > 0.5
                    val_correct += (predictions == labels.bool()).sum().item()
                    val_total += labels.size(0)
                    
                    val_bar.set_postfix({
                        'loss': loss.item(),
                        'acc': val_correct / val_total
                    })
            
            avg_val_loss = val_loss / len(val_loader)
            val_acc = val_correct / val_total
            
            log.info(f'Epoch {epoch+1}/{epochs}:')
            log.info(f'  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}')
            log.info(f'  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}')
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0

                best_model_state = copy.deepcopy(model.state_dict())
                log.info(f"  新的最佳验证损失: {best_val_loss:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    log.info(f'Early stopping at epoch {epoch+1}')
                    break
    
    model.load_state_dict(best_model_state)
    log.info("模型训练完成!")
    


def main():
    # 数据路径设置
    if args.mode == 'valid':
        data_path = '/home/wangtiantian/shikainan/newscommemder/top2wk/user_data/data/offline'
        model_dir = '/home/wangtiantian/shikainan/newscommemder/top2wk/user_data/model/offline'
    else:
        data_path = '/home/wangtiantian/shikainan/newscommemder/top2wk/user_data/data/online'  
        model_dir = '/home/wangtiantian/shikainan/newscommemder/top2wk/user_data/model/online'
    
    os.makedirs(model_dir, exist_ok=True)
    
    # 模型和编码器保存路径
    model_path = os.path.join(model_dir, 'youtube_dnn.pth')
    encoders_path = os.path.join(model_dir, 'encoders.pkl')
    config_path = os.path.join(model_dir, 'model_config.pkl')
    
    # 读取数据
    click_path = os.path.join(data_path, 'click.pkl')
    if not os.path.exists(click_path):
        log.error(f"数据文件不存在: {click_path}")
        return
    
    df_click = pd.read_pickle(click_path)
    log.info(f'点击数据形状: {df_click.shape}')
    log.debug(f'点击数据示例:\n{df_click.head()}')
    
    # 准备训练数据
    train_data, val_data, user_encoder, item_encoder = prepare_training_data(
        df_click, 
        test_ratio=args.test_ratio,
        max_seq_len=args.max_seq_len
    )
    
    # 创建数据加载器
    train_dataset = YouTubeDNNDataset(*train_data)
    val_dataset = YouTubeDNNDataset(*val_data)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers
    )
    
    # 构建模型
    model = YouTubeDNN(
        user_num=len(user_encoder.classes_),
        item_num=len(item_encoder.classes_),
        embedding_dim=args.embedding_dim,
        hidden_units=args.hidden_units
    )
    
    log.info("模型结构:")
    log.info(f"  用户数: {len(user_encoder.classes_)}")
    log.info(f"  物品数: {len(item_encoder.classes_)}")
    log.info(f"  Embedding维度: {args.embedding_dim}")
    log.info(f"  隐藏层: {args.hidden_units}")
    log.info(f"  最大序列长度: {args.max_seq_len}")
    
    # 训练模型
    train_model(
        model, 
        train_loader, 
        val_loader, 
        args.epochs, 
        args.learning_rate, 
        args.patience
    )
    
    # 保存模型、编码器和配置
    log.info("保存模型...")
    torch.save(model.state_dict(), model_path)
    
    with open(encoders_path, 'wb') as f:
        pickle.dump((user_encoder, item_encoder), f)
    
    # 保存模型配置，供召回时使用
    model_config = {
        'user_num': len(user_encoder.classes_),
        'item_num': len(item_encoder.classes_),
        'embedding_dim': args.embedding_dim,
        'hidden_units': args.hidden_units,
        'max_seq_len': args.max_seq_len,
    }
    
    with open(config_path, 'wb') as f:
        pickle.dump(model_config, f)
    
    log.info(f"模型保存到: {model_path}")
    log.info(f"编码器保存到: {encoders_path}")
    log.info(f"配置保存到: {config_path}")
    log.info("训练完成!")


if __name__ == '__main__':
    main()