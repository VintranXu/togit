import argparse
import gc
import os
import random
import warnings
from collections import defaultdict
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import joblib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import Logger, evaluate, gen_sub

# 导入DIEN模型
from model.DIEN import DIEN

warnings.filterwarnings('ignore')

seed = 2020
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# 命令行参数
parser = argparse.ArgumentParser(description='DIEN 精排模型')
parser.add_argument('--mode', default='valid')
parser.add_argument('--logfile', default='dien_ranking.log')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--max_seq_len', type=int, default=50)
parser.add_argument('--embedding_dim', type=int, default=128)
parser.add_argument('--hidden_dim', type=int, default=128)

# 测试模式参数
parser.add_argument('--test_mode', action='store_true', help='使用少量数据进行快速测试')
parser.add_argument('--test_samples', type=int, default=5000, help='测试模式下使用的样本数量')

args = parser.parse_args()

mode = args.mode
logfile = args.logfile

# 初始化日志
os.makedirs('./user_data/log', exist_ok=True)
log = Logger(f'./user_data/log/{logfile}').logger
log.info(f'DIEN 精排训练，mode: {mode}')

# 设置设备
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
log.info(f'Using device: {device}')


class RealNewsDataset(Dataset):
    """真正的DIEN新闻推荐数据集 - 使用真实的用户历史序列"""
    def __init__(self, features, labels, user_history, user_encoders=None, max_seq_len=50):
        self.features = features
        self.labels = labels
        self.user_history = user_history
        self.user_encoders = user_encoders
        self.max_seq_len = max_seq_len
        
        # 预计算数值特征
        self.numeric_feature_cols = [
            'words_count', 
            'user_id_click_article_created_at_ts_diff_mean',
            'user_id_click_diff_mean',
            'user_click_timestamp_created_at_ts_diff_mean',
            'user_click_timestamp_created_at_ts_diff_std',
            'user_click_datetime_hour_std',
            'user_clicked_article_words_count_mean',
            'user_last_click_created_at_ts_diff',
            'user_last_click_timestamp_diff',
            'user_last_click_words_count_diff'
        ]
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        # 获取特征
        feature = self.features.iloc[idx]
        label = self.labels[idx] if self.labels is not None else 0
        
        # 基础特征
        user_id = int(feature['user_id'])
        article_id = int(feature['article_id'])
        category_id = int(feature['category_id']) if 'category_id' in feature else 0
        
        # 构建真实的用户历史序列
        hist_items = []
        hist_cats = []
        seq_len = 0
        
        if user_id in self.user_history:
            # 获取用户历史
            user_hist = self.user_history[user_id]
            hist_items_raw = user_hist.get('items', [])
            hist_cats_raw = user_hist.get('categories', [])
            
            # 取最近的历史记录
            if len(hist_items_raw) > 0:
                # 取最近的max_seq_len个交互
                hist_items = hist_items_raw[-self.max_seq_len:]
                hist_cats = hist_cats_raw[-self.max_seq_len:]
                seq_len = len(hist_items)
                
                # 如果序列长度不足，进行padding
                if seq_len < self.max_seq_len:
                    pad_len = self.max_seq_len - seq_len
                    hist_items = [0] * pad_len + hist_items
                    hist_cats = [0] * pad_len + hist_cats
                else:
                    seq_len = self.max_seq_len
        
        # 如果没有历史记录，用0填充
        if len(hist_items) == 0:
            hist_items = [0] * self.max_seq_len
            hist_cats = [0] * self.max_seq_len
            seq_len = 0
        
        # 获取数值特征
        numeric_features = []
        for col in self.numeric_feature_cols:
            if col in feature:
                val = feature[col]
                if pd.isna(val):
                    numeric_features.append(0.0)
                else:
                    numeric_features.append(float(val))
            else:
                numeric_features.append(0.0)
                
        return {
            'user_id': user_id,
            'article_id': article_id,
            'category_id': category_id,
            'hist_items': torch.LongTensor(hist_items),
            'hist_cats': torch.LongTensor(hist_cats),
            'seq_len': seq_len,
            'features': torch.FloatTensor(numeric_features),
            'label': torch.FloatTensor([label])
        }


def prepare_user_history(df_click, df_feature):
    """构建用户历史交互数据"""
    log.info("构建用户历史交互数据...")
    
    # 合并点击数据和特征数据以获取category信息
    df_click_with_cat = df_click.merge(
        df_feature[['article_id', 'category_id']].drop_duplicates(),
        left_on='click_article_id',
        right_on='article_id',
        how='left'
    )
    
    # 构建用户历史字典
    user_history = {}
    
    # 按用户分组并构建历史序列
    for user_id, group in tqdm(df_click_with_cat.groupby('user_id'), desc="构建用户历史"):
        # 按时间排序
        group = group.sort_values('click_timestamp')
        
        user_history[user_id] = {
            'items': group['click_article_id'].tolist(),
            'categories': group['category_id'].fillna(0).astype(int).tolist(),
            'timestamps': group['click_timestamp'].tolist()
        }
    
    log.info(f"构建了 {len(user_history)} 个用户的历史数据")
    
    # 统计历史长度分布
    hist_lengths = [len(hist['items']) for hist in user_history.values()]
    log.info(f"历史长度统计: min={min(hist_lengths)}, max={max(hist_lengths)}, "
            f"mean={np.mean(hist_lengths):.2f}, median={np.median(hist_lengths):.2f}")
    
    return user_history


def prepare_features_and_scalers(df_feature):
    """准备特征并进行标准化"""
    log.info("准备数值特征并标准化...")
    
    # 数值特征列
    numeric_cols = [
        'sim_score', 'created_at_ts', 'words_count',
        'user_id_click_article_created_at_ts_diff_mean',
        'user_id_click_diff_mean',
        'user_click_timestamp_created_at_ts_diff_mean',
        'user_click_timestamp_created_at_ts_diff_std',
        'user_click_datetime_hour_std',
        'user_clicked_article_words_count_mean',
        'user_last_click_created_at_ts_diff',
        'user_last_click_timestamp_diff',
        'user_last_click_words_count_diff'
    ]
    
    # 过滤存在的列
    existing_numeric_cols = [col for col in numeric_cols if col in df_feature.columns]
    log.info(f"可用的数值特征: {existing_numeric_cols}")
    
    # 标准化数值特征
    scaler = StandardScaler()
    numeric_features = df_feature[existing_numeric_cols].fillna(0).values
    numeric_features_scaled = scaler.fit_transform(numeric_features)
    
    # 将标准化后的特征添加回DataFrame
    for i, col in enumerate(existing_numeric_cols):
        df_feature[f'{col}_scaled'] = numeric_features_scaled[:, i]
    
    return df_feature, scaler, existing_numeric_cols


def train_model(df_feature, df_query, df_click):
    """训练真正的DIEN模型"""
    log.info("开始训练真正的DIEN精排模型...")
    
    # 测试模式：采样少量数据
    if args.test_mode:
        log.info(f"测试模式：采样 {args.test_samples} 条数据")
        if len(df_feature) > args.test_samples:
            df_feature = df_feature.sample(n=args.test_samples, random_state=42).reset_index(drop=True)
            # 相应地过滤点击数据
            valid_users = set(df_feature['user_id'].unique())
            df_click = df_click[df_click['user_id'].isin(valid_users)]
            log.info(f"采样后特征数据大小: {len(df_feature)}, 点击数据大小: {len(df_click)}")
    
    # 1. 构建用户历史数据
    user_history = prepare_user_history(df_click, df_feature)
    
    # 2. 准备和标准化特征
    df_feature, scaler, numeric_cols = prepare_features_and_scalers(df_feature)
    
    # 3. 根据label是否为空，切分训练集和测试集
    df_train = df_feature[df_feature['label'].notnull()].copy()
    df_test = df_feature[df_feature['label'].isnull()].copy()
    
    log.info(f'训练集大小: {len(df_train)}, 测试集大小: {len(df_test)}')
    
    # 4. 对类别特征进行编码
    categorical_features = ['user_id', 'article_id', 'category_id']
    label_encoders = {}
    
    for feat in categorical_features:
        if feat in df_train.columns:
            le = LabelEncoder()
            # 合并训练集和测试集的数据进行编码
            all_values = pd.concat([df_train[feat], df_test[feat]]).astype(str)
            le.fit(all_values)
            df_train[feat] = le.transform(df_train[feat].astype(str))
            df_test[feat] = le.transform(df_test[feat].astype(str))
            label_encoders[feat] = le
            
            # 同时编码用户历史数据
            if feat in ['article_id']:  # 只需要编码物品ID，用户ID已经是键
                for user_id in user_history:
                    if feat == 'article_id':
                        # 编码历史物品ID
                        try:
                            encoded_items = le.transform([str(item) for item in user_history[user_id]['items']])
                            user_history[user_id]['items'] = encoded_items.tolist()
                        except ValueError:
                            # 如果历史中有训练集没见过的物品，设为0
                            encoded_items = []
                            for item in user_history[user_id]['items']:
                                try:
                                    encoded_items.append(le.transform([str(item)])[0])
                                except ValueError:
                                    encoded_items.append(0)  # 未知物品
                            user_history[user_id]['items'] = encoded_items
    
    # 5. 获取编码后的维度
    n_users = max(df_train['user_id'].max(), df_test['user_id'].max()) + 1
    n_items = max(df_train['article_id'].max(), df_test['article_id'].max()) + 1
    n_categories = max(df_train['category_id'].max(), df_test['category_id'].max()) + 1 if 'category_id' in df_train.columns else 1
    
    log.info(f'Number of users: {n_users}')
    log.info(f'Number of items: {n_items}')
    log.info(f'Number of categories: {n_categories}')
    
    # 6. 准备标签
    ycol = 'label'
    y_train = df_train[ycol].values
    
    # 7. 初始化预测结果
    oof = []
    prediction = df_test[['user_id', 'article_id']].copy()
    prediction['pred'] = 0
    
    # 8. 5折交叉验证
    kfold = GroupKFold(n_splits=5)
    
    for fold_id, (trn_idx, val_idx) in enumerate(
            kfold.split(df_train, y_train, df_train['user_id'])):
        
        log.info(f'\nFold_{fold_id + 1} Training ================================\n')
        
        # 准备训练和验证数据
        X_train = df_train.iloc[trn_idx]
        y_train_fold = y_train[trn_idx]
        X_val = df_train.iloc[val_idx]
        y_val_fold = y_train[val_idx]
        
        # 创建数据集 - 使用真实的用户历史
        train_dataset = RealNewsDataset(X_train, y_train_fold, user_history, 
                                      label_encoders, args.max_seq_len)
        val_dataset = RealNewsDataset(X_val, y_val_fold, user_history, 
                                    label_encoders, args.max_seq_len)
        test_dataset = RealNewsDataset(df_test, None, user_history, 
                                     label_encoders, args.max_seq_len)
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                                shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                              shuffle=False, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                                shuffle=False, num_workers=4, pin_memory=True)
        
        # 初始化模型
        model = DIEN(
            n_users=n_users,
            n_items=n_items,
            n_categories=n_categories,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            num_features=len(train_dataset.numeric_feature_cols)
        ).to(device)
        
        # 损失函数和优化器
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5)
        
        # 训练模型
        best_auc = 0
        early_stop_counter = 0
        early_stop_patience = 5
        
        for epoch in range(args.epochs):
            # 训练阶段
            model.train()
            train_loss = 0
            train_preds = []
            train_labels = []
            
            for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs} - Training'):
                user_ids = batch['user_id'].to(device)
                article_ids = batch['article_id'].to(device)
                category_ids = batch['category_id'].to(device)
                hist_items = batch['hist_items'].to(device)
                hist_cats = batch['hist_cats'].to(device)
                seq_lens = batch['seq_len'].to(device)
                features = batch['features'].to(device)
                labels = batch['label'].to(device)
                
                optimizer.zero_grad()
                outputs = model(user_ids, article_ids, category_ids, 
                              hist_items, hist_cats, seq_lens, features)
                loss = criterion(outputs, labels)
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                
                train_loss += loss.item()
                train_preds.extend(outputs.cpu().detach().numpy().flatten())
                train_labels.extend(labels.cpu().detach().numpy().flatten())
            
            # 验证阶段
            model.eval()
            val_loss = 0
            val_preds = []
            val_labels = []
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f'Epoch {epoch+1}/{args.epochs} - Validation'):
                    user_ids = batch['user_id'].to(device)
                    article_ids = batch['article_id'].to(device)
                    category_ids = batch['category_id'].to(device)
                    hist_items = batch['hist_items'].to(device)
                    hist_cats = batch['hist_cats'].to(device)
                    seq_lens = batch['seq_len'].to(device)
                    features = batch['features'].to(device)
                    labels = batch['label'].to(device)
                    
                    outputs = model(user_ids, article_ids, category_ids,
                                  hist_items, hist_cats, seq_lens, features)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    val_preds.extend(outputs.cpu().detach().numpy().flatten())
                    val_labels.extend(labels.cpu().detach().numpy().flatten())
            
            # 计算AUC
            train_labels_array = np.array(train_labels)
            val_labels_array = np.array(val_labels)
            
            if len(np.unique(train_labels_array)) > 1:
                train_auc = roc_auc_score(train_labels_array, train_preds)
            else:
                train_auc = 0.5
                
            if len(np.unique(val_labels_array)) > 1:
                val_auc = roc_auc_score(val_labels_array, val_preds)
            else:
                val_auc = 0.5
            
            log.info(f'Epoch {epoch+1}: Train Loss={train_loss/len(train_loader):.4f}, '
                    f'Train AUC={train_auc:.4f}, Val Loss={val_loss/len(val_loader):.4f}, '
                    f'Val AUC={val_auc:.4f}')
            
            # 学习率调整
            scheduler.step(val_auc)
            
            # Early stopping
            if val_auc > best_auc:
                best_auc = val_auc
                early_stop_counter = 0
                # 保存最佳模型
                torch.save(model.state_dict(), 
                          f'./user_data/model/real_dien_fold{fold_id}.pth')
            else:
                early_stop_counter += 1
                if early_stop_counter >= early_stop_patience:
                    log.info(f'Early stopping at epoch {epoch+1}')
                    break
        
        # 加载最佳模型进行预测
        model.load_state_dict(torch.load(
            f'./user_data/model/real_dien_fold{fold_id}.pth'))
        model.eval()
        
        # 验证集预测
        val_preds = []
        with torch.no_grad():
            for batch in val_loader:
                user_ids = batch['user_id'].to(device)
                article_ids = batch['article_id'].to(device)
                category_ids = batch['category_id'].to(device)
                hist_items = batch['hist_items'].to(device)
                hist_cats = batch['hist_cats'].to(device)
                seq_lens = batch['seq_len'].to(device)
                features = batch['features'].to(device)
                
                outputs = model(user_ids, article_ids, category_ids,
                              hist_items, hist_cats, seq_lens, features)
                val_preds.extend(outputs.cpu().detach().numpy())
        
        # 保存OOF预测
        df_oof = X_val[['user_id', 'article_id']].copy()
        df_oof['label'] = y_val_fold
        df_oof['pred'] = np.array(val_preds).flatten()
        oof.append(df_oof)
        
        # 测试集预测
        test_preds = []
        with torch.no_grad():
            for batch in test_loader:
                user_ids = batch['user_id'].to(device)
                article_ids = batch['article_id'].to(device)
                category_ids = batch['category_id'].to(device)
                hist_items = batch['hist_items'].to(device)
                hist_cats = batch['hist_cats'].to(device)
                seq_lens = batch['seq_len'].to(device)
                features = batch['features'].to(device)
                
                outputs = model(user_ids, article_ids, category_ids,
                              hist_items, hist_cats, seq_lens, features)
                test_preds.extend(outputs.cpu().detach().numpy())
        
        prediction['pred'] += np.array(test_preds).flatten() / 5
        
        # 保存相关数据
        joblib.dump({
            'label_encoders': label_encoders,
            'scaler': scaler,
            'numeric_cols': numeric_cols,
            'user_history_sample': dict(list(user_history.items())[:100])  # 保存部分历史作为样例
        }, f'./user_data/model/real_dien_metadata_fold{fold_id}.pkl')
        
        # 清理内存
        del model, train_dataset, val_dataset, test_dataset
        del train_loader, val_loader, test_loader
        torch.cuda.empty_cache()
        gc.collect()
    
    # 生成线下结果
    df_oof = pd.concat(oof)
    df_oof.sort_values(['user_id', 'pred'], inplace=True, ascending=[True, False])
    log.info(f'df_oof.head: {df_oof.head()}')
    
    # 计算相关指标
    total = df_query[df_query['click_article_id'] != -1].user_id.nunique()
    hitrate_5, mrr_5, hitrate_10, mrr_10, hitrate_20, mrr_20, hitrate_40, mrr_40, hitrate_50, mrr_50 = evaluate(
        df_oof, total)
    log.info(f'DIEN精排指标: HR@5={hitrate_5:.4f}, MRR@5={mrr_5:.4f}, HR@10={hitrate_10:.4f}, '
            f'MRR@10={mrr_10:.4f}, HR@20={hitrate_20:.4f}, MRR@20={mrr_20:.4f}, '
            f'HR@40={hitrate_40:.4f}, MRR@40={mrr_40:.4f}, HR@50={hitrate_50:.4f}, MRR@50={mrr_50:.4f}')
    
    # 生成提交文件
    df_sub = gen_sub(prediction)
    df_sub.sort_values(['user_id'], inplace=True)
    os.makedirs('./prediction_result', exist_ok=True)
    df_sub.to_csv(f'./prediction_result/real_dien_result.csv', index=False)
    log.info('DIEN精排结果保存完成!')


if __name__ == '__main__':
    # 确保模型保存目录存在
    os.makedirs('./user_data/model', exist_ok=True)
    
    if mode == 'valid':
        # 加载特征数据、查询数据和点击数据
        df_feature = pd.read_pickle('./user_data/data/offline/feature.pkl')
        df_query = pd.read_pickle('./user_data/data/offline/query.pkl')
        df_click = pd.read_pickle('./user_data/data/offline/click.pkl')
        
        log.info(f"加载数据: 特征={df_feature.shape}, 查询={df_query.shape}, 点击={df_click.shape}")
        
        # 找出所有包含文本数据的列进行编码（除了我们要特殊处理的）
        text_columns = [col for col in df_feature.select_dtypes('object').columns 
                       if col not in ['user_id', 'article_id', 'category_id']]
        for f in text_columns:
            lbl = LabelEncoder()
            df_feature[f] = lbl.fit_transform(df_feature[f].astype(str))
        
        # 训练真正的DIEN模型
        train_model(df_feature, df_query, df_click)
        
    else:
        log.error("目前只支持 valid 模式的DIEN精排训练")