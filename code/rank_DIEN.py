import argparse
import gc
import os
import random
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm

from model.DIEN import DIEN
from data_provider.DIENDataset import NewsDataset
from torch.utils.data import DataLoader, Dataset

from utils import Logger, evaluate, gen_sub

warnings.filterwarnings('ignore')


#随机种子设置
seed = 2020
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# 命令行参数
parser = argparse.ArgumentParser(description='DIEN 排序')
parser.add_argument('--mode', default='valid')
parser.add_argument('--logfile', default='test.log')
parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--embedding_dim', type=int, default=128)
parser.add_argument('--hidden_dim', type=int, default=128)

args = parser.parse_args()

mode = args.mode
logfile = args.logfile
device = torch.device(args.device)

# 初始化日志
os.makedirs('../user_data/log', exist_ok=True)
log = Logger(f'../user_data/log/{logfile}').logger
log.info(f'DIEN 排序，mode: {mode}, device: {device}')








# =========================== 训练和预测函数 ===========================
def prepare_data(df_feature, df_click, df_query):
    """准备数据"""
    # 构建用户历史字典
    user_history = {}
    
    #构建用户的点击字典：每个用户的历史包含物品、类别、时间戳三个维度
    for user_id, group in df_click.groupby('user_id'):
        group = group.sort_values('click_timestamp')
        user_history[user_id] = {
            'items': group['click_article_id'].tolist(),
            'categories': group['category_id'].fillna(0).astype(int).tolist(),
            'timestamps': group['click_timestamp'].tolist()
        }
    
    # 准备要处理的特征
    numeric_cols = [
        'sim_score', 'created_at_ts', 'words_count',
        'user_id_click_article_created_at_ts_diff_mean',
        'user_id_click_diff_mean',
        'user_click_timestamp_created_at_ts_diff_mean',
        'user_click_timestamp_created_at_ts_diff_std',
        'user_click_datetime_hour_std',
        'user_clicked_article_words_count_mean',
        'user_click_last_article_words_count'
    ]
    
    # 标准化数值特征
    scaler = StandardScaler()
    # 将缺失值填充为0后转为numpy数组，防止后续计算出错
    numeric_features = df_feature[numeric_cols].fillna(0).values
    #标准化特征 (sample_num, feature_num)
    numeric_features = scaler.fit_transform(numeric_features)
    
    # 构建特征字典
    features = []
    for idx, row in df_feature.iterrows():
        features.append({
            'user_id': row['user_id'],
            'article_id': row['article_id'],
            'category_id': row['category_id'] if not pd.isna(row['category_id']) else 0,
            'numeric_features': numeric_features[idx]
        })
    
    """
    features: 每个样本的若干特征 {user_id: 0, article_id:1205, category_id:2, numeric_faetures: [sim_score, ...., user_click_last_article_words_count] }
    user_history: 用户的交互特征 {user_id: {items: [1, 2, 3, ...], categories: [....], timestamp: [.....]}}
    scaler: 确保训练和预测时使用相同的特征标准化方式
    """
    return features, user_history, scaler


def train_model(df_feature, df_query, df_click):
    """训练DIEN模型"""
    # 准备数据， label不是null代表是验证集的内容，如果是null则代表测试集的内容，要计算后最终保留输出
    df_train = df_feature[df_feature['label'].notnull()].copy()
    df_test = df_feature[df_feature['label'].isnull()].copy()
    
    # 获取实体数量
    n_users = df_feature['user_id'].max() + 1
    n_items = df_feature['article_id'].max() + 1
    n_categories = df_feature['category_id'].fillna(0).astype(int).max() + 1
    
    log.info(f'n_users: {n_users}, n_items: {n_items}, n_categories: {n_categories}')
    
    # 准备特征
    train_features, user_history, scaler = prepare_data(df_train, df_click, df_query)
    test_features, _, _ = prepare_data(df_test, df_click, df_query)
    
    # 训练集的label取出，代表是不是真正的结果
    train_labels = df_train['label'].values
    
    # 5折交叉验证
    kfold = GroupKFold(n_splits=5)
    oof_predictions = []
    test_predictions = []
    
    for fold_id, (trn_idx, val_idx) in enumerate(
            kfold.split(df_train, train_labels, df_train['user_id'])):
        
        log.info(f'\nFold_{fold_id + 1} Training ================================\n')
        
        # 准备训练和验证数据
        X_train = [train_features[i] for i in trn_idx]
        y_train = train_labels[trn_idx]
        X_val = [train_features[i] for i in val_idx]
        y_val = train_labels[val_idx]
        
        # 创建数据集
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
        train_dataset = NewsDataset(X_train, y_train, user_history)
        val_dataset = NewsDataset(X_val, y_val, user_history)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                                shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                               shuffle=False, num_workers=4)
        
        # 创建模型
        # num_features： 标准化后的数值特征的维度
        model = DIEN(n_users, n_items, n_categories, 
                    embedding_dim=args.embedding_dim,
                    hidden_dim=args.hidden_dim,
                    num_features=len(train_features[0]['numeric_features']))
        model = model.to(device)
        
        # 优化器和损失函数
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        criterion = nn.BCELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=2, factor=0.5
        )
        
        # 训练模型
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(args.epochs):
            # 训练阶段
            model.train()
            train_loss = 0
            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs} [Train]')
            
            for batch in train_pbar:
                # 数据移到设备
                user_ids = batch['user_id'].to(device)
                article_ids = batch['article_id'].to(device)
                category_ids = batch['category_id'].to(device)
                hist_items = batch['hist_items'].to(device)
                hist_cats = batch['hist_cats'].to(device)
                seq_lens = batch['seq_len'].to(device)
                features = batch['features'].to(device)
                labels = batch['label'].to(device)
                
                # 前向传播
                optimizer.zero_grad()
                outputs = model(user_ids, article_ids, category_ids,
                              hist_items, hist_cats, seq_lens, features)
                loss = criterion(outputs, labels)
                
                # 反向传播
                loss.backward()
                #进行梯度裁剪，防止出现梯度爆炸
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                
                train_loss += loss.item()
                train_pbar.set_postfix({'loss': loss.item()})
            
            # 验证阶段
            model.eval()
            val_loss = 0
            val_preds = []
            val_labels = []
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f'Epoch {epoch+1}/{args.epochs} [Valid]'):
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
                    val_preds.extend(outputs.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())
            
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            log.info(f'Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}')
            
            # 学习率调整
            scheduler.step(avg_val_loss)
            
            # 早停
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # 保存模型
                torch.save(model.state_dict(), f'../user_data/model/dien_fold{fold_id}.pth')
            else:
                patience_counter += 1
                if patience_counter >= 5:
                    log.info(f'Early stopping at epoch {epoch+1}')
                    break
        
        # 加载最佳模型
        model.load_state_dict(torch.load(f'../user_data/model/dien_fold{fold_id}.pth'))
        
        # 验证集预测
        model.eval()
        val_preds = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Final Validation'):
                user_ids = batch['user_id'].to(device)
                article_ids = batch['article_id'].to(device)
                category_ids = batch['category_id'].to(device)
                hist_items = batch['hist_items'].to(device)
                hist_cats = batch['hist_cats'].to(device)
                seq_lens = batch['seq_len'].to(device)
                features = batch['features'].to(device)
                
                outputs = model(user_ids, article_ids, category_ids,
                              hist_items, hist_cats, seq_lens, features)
                val_preds.extend(outputs.cpu().numpy())
        
        # 保存OOF预测
        df_oof = df_train.iloc[val_idx][['user_id', 'article_id', 'label']].copy()
        df_oof['pred'] = np.array(val_preds).flatten()
        oof_predictions.append(df_oof)
        
        # 测试集预测
        test_dataset = NewsDataset(test_features, None, user_history)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                               shuffle=False, num_workers=4)
        
        test_preds = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc='Test Prediction'):
                user_ids = batch['user_id'].to(device)
                article_ids = batch['article_id'].to(device)
                category_ids = batch['category_id'].to(device)
                hist_items = batch['hist_items'].to(device)
                hist_cats = batch['hist_cats'].to(device)
                seq_lens = batch['seq_len'].to(device)
                features = batch['features'].to(device)
                
                outputs = model(user_ids, article_ids, category_ids,
                              hist_items, hist_cats, seq_lens, features)
                test_preds.extend(outputs.cpu().numpy())
        
        test_predictions.append(np.array(test_preds).flatten())
        
        # 清理内存
        del model
        torch.cuda.empty_cache()
        gc.collect()
    
    # 合并OOF预测
    df_oof = pd.concat(oof_predictions)
    df_oof.sort_values(['user_id', 'pred'], inplace=True, ascending=[True, False])
    log.info(f'df_oof.head: {df_oof.head()}')
    
    # 计算指标
    total = df_query[df_query['click_article_id'] != -1].user_id.nunique()
    hitrate_5, mrr_5, hitrate_10, mrr_10, hitrate_20, mrr_20, hitrate_40, mrr_40, hitrate_50, mrr_50 = evaluate(
        df_oof, total)
    log.info(
        f'Metrics: HR@5={hitrate_5:.4f}, MRR@5={mrr_5:.4f}, '
        f'HR@10={hitrate_10:.4f}, MRR@10={mrr_10:.4f}, '
        f'HR@20={hitrate_20:.4f}, MRR@20={mrr_20:.4f}'
    )
    
    # 生成提交文件
    prediction = df_test[['user_id', 'article_id']].copy()
    prediction['pred'] = np.mean(test_predictions, axis=0)
    
    df_sub = gen_sub(prediction)
    df_sub.sort_values(['user_id'], inplace=True)
    os.makedirs('../prediction_result', exist_ok=True)
    df_sub.to_csv('../prediction_result/result_dien.csv', index=False)
    
    # 保存scaler
    joblib.dump(scaler, '../user_data/model/dien_scaler.pkl')


def online_predict(df_test, df_click):
    """在线预测"""
    # 获取实体数量
    n_users = df_test['user_id'].max() + 1
    n_items = df_test['article_id'].max() + 1
    n_categories = df_test['category_id'].fillna(0).astype(int).max() + 1
    
    # 准备特征
    test_features, user_history, _ = prepare_data(df_test, df_click, None)
    
    # 加载scaler
    scaler = joblib.load('../user_data/model/dien_scaler.pkl')
    
    # 创建测试数据集
    test_dataset = NewsDataset(test_features, None, user_history)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=4)
    
    # 多折预测
    all_predictions = []
    
    for fold_id in range(5):
        log.info(f'Loading model fold {fold_id}')
        
        # 创建模型
        model = DIEN(n_users, n_items, n_categories,
                    embedding_dim=args.embedding_dim,
                    hidden_dim=args.hidden_dim,
                    num_features=len(test_features[0]['numeric_features']))
        
        # 加载权重
        model.load_state_dict(torch.load(f'../user_data/model/dien_fold{fold_id}.pth'))
        model = model.to(device)
        model.eval()
        
        # 预测
        fold_preds = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f'Fold {fold_id} Prediction'):
                user_ids = batch['user_id'].to(device)
                article_ids = batch['article_id'].to(device)
                category_ids = batch['category_id'].to(device)
                hist_items = batch['hist_items'].to(device)
                hist_cats = batch['hist_cats'].to(device)
                seq_lens = batch['seq_len'].to(device)
                features = batch['features'].to(device)
                
                outputs = model(user_ids, article_ids, category_ids,
                              hist_items, hist_cats, seq_lens, features)
                fold_preds.extend(outputs.cpu().numpy())
        
        all_predictions.append(np.array(fold_preds).flatten())
        
        # 清理内存
        del model
        torch.cuda.empty_cache()
    
    # 融合预测
    prediction = df_test[['user_id', 'article_id']].copy()
    prediction['pred'] = np.mean(all_predictions, axis=0)
    
    # 生成提交文件
    df_sub = gen_sub(prediction)
    df_sub.sort_values(['user_id'], inplace=True)
    os.makedirs('../prediction_result', exist_ok=True)
    df_sub.to_csv('../prediction_result/result_dien.csv', index=False)


if __name__ == '__main__':
    # 创建模型保存目录
    os.makedirs('../user_data/model', exist_ok=True)
    
    if mode == 'valid':
        # 加载数据
        df_feature = pd.read_pickle('../user_data/data/offline/feature.pkl')
        df_query = pd.read_pickle('../user_data/data/offline/query.pkl')
        df_click = pd.read_pickle('../user_data/data/offline/click.pkl')
        
        # 处理类别特征
        df_click = df_click.merge(
            df_feature[['article_id', 'category_id']].drop_duplicates(),
            left_on='click_article_id',
            right_on='article_id',
            how='left'
        )
        
        # 编码处理
        for f in ['user_id', 'article_id', 'category_id']:
            if f in df_feature.columns:
                lbl = LabelEncoder()
                df_feature[f] = lbl.fit_transform(df_feature[f].fillna('nan').astype(str))
                if f in df_click.columns:
                    df_click[f] = lbl.transform(df_click[f].fillna('nan').astype(str))
        
        # 训练模型
        train_model(df_feature, df_query, df_click)
        
    else:
        # 在线预测
        df_feature = pd.read_pickle('../user_data/data/online/feature.pkl')
        df_click = pd.read_pickle('../user_data/data/online/click.pkl')
        
        # 处理类别特征
        df_click = df_click.merge(
            df_feature[['article_id', 'category_id']].drop_duplicates(),
            left_on='click_article_id',
            right_on='article_id',
            how='left'
        )
        
        # 编码处理
        for f in ['user_id', 'article_id', 'category_id']:
            if f in df_feature.columns:
                lbl = LabelEncoder()
                df_feature[f] = lbl.fit_transform(df_feature[f].fillna('nan').astype(str))
                if f in df_click.columns:
                    df_click