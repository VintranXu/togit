import argparse
import os
import pickle
import random
import sys
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from annoy import AnnoyIndex

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.YouTubeDNNModel import YouTubeDNN
from utils import Logger, evaluate

warnings.filterwarnings('ignore')

# 设置随机种子
seed = 2020
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

# 命令行参数
parser = argparse.ArgumentParser(description='YouTube DNN模型召回')
parser.add_argument('--mode', default='valid', choices=['valid', 'online'],
                    help='召回模式：valid(离线验证) 或 online(在线)')
parser.add_argument('--logfile', default='recall_youtube_dnn.log', help='日志文件名')
parser.add_argument('--batch_size', default=512, type=int, help='推理批大小')
parser.add_argument('--top_k', default=100, type=int, help='每用户召回物品数')
parser.add_argument('--recall_num', default=50, type=int, help='最终召回物品数')

args = parser.parse_args()

# 设备选择
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 初始化日志
os.makedirs('/home/wangtiantian/shikainan/newscommemder/top2wk/user_data/log', exist_ok=True)
log = Logger(f'/home/wangtiantian/shikainan/newscommemder/top2wk/user_data/log/{args.logfile}').logger
log.info(f'YouTube DNN召回开始，mode: {args.mode}, device: {device}')
log.info(f'召回参数: {vars(args)}')


def load_model_and_encoders(model_dir):
    """加载训练好的模型和编码器"""
    model_path = os.path.join(model_dir, 'youtube_dnn.pth')
    encoders_path = os.path.join(model_dir, 'encoders.pkl')
    config_path = os.path.join(model_dir, 'model_config.pkl')
    
    # 检查文件是否存在
    if not all(os.path.exists(p) for p in [model_path, encoders_path, config_path]):
        missing = [p for p in [model_path, encoders_path, config_path] if not os.path.exists(p)]
        raise FileNotFoundError(f"缺少模型文件: {missing}")
    
    # 加载配置
    with open(config_path, 'rb') as f:
        model_config = pickle.load(f)
    
    # 加载编码器
    with open(encoders_path, 'rb') as f:
        user_encoder, item_encoder = pickle.load(f)
    
    # 重建模型
    model = YouTubeDNN(
        user_num=model_config['user_num'],
        item_num=model_config['item_num'],
        embedding_dim=model_config['embedding_dim'],
        hidden_units=model_config['hidden_units']
    )
    
    # 加载模型权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    log.info("模型加载成功!")
    log.info(f"用户数: {model_config['user_num']}")
    log.info(f"物品数: {model_config['item_num']}")
    log.info(f"最大序列长度: {model_config['max_seq_len']}")
    
    return model, user_encoder, item_encoder, model_config


def build_user_item_mappings(df_click):
    """构建用户-物品交互映射"""
    user_item_dict = df_click.groupby('user_id')['click_article_id'].agg(list).to_dict()
    return user_item_dict




def batch_recall_optimized(df_query, model, user_encoder, item_encoder, 
                          user_item_dict, model_config, batch_size=512, 
                          top_k=100, recall_num=50):
    """优化的批量GPU召回"""
    log.info("开始批量GPU推理...")
    model.eval()
    
    max_seq_len = model_config['max_seq_len']
    embedding_dim = model_config['embedding_dim']

    all_results = []
    
    # 1. 预计算所有物品embedding（一次性完成）
    log.info("预计算物品embedding...")
    with torch.no_grad():
        all_items = torch.LongTensor(np.arange(1, len(item_encoder.classes_) + 1)).to(device)
        all_item_embeddings = model.get_item_embeddings(all_items)  # 保持在GPU上
        log.info(f"物品embedding形状: {all_item_embeddings.shape}")
        #转换到cpu上来构建
        all_item_embeddings_cpu = all_item_embeddings.cpu().numpy()
        log.info(f"物品embedding形状: {all_item_embeddings_cpu.shape}")
    
    # 2. 构建AnnoyIndex
    log.info("构建AnnoyIndex...")
    annoy_index = AnnoyIndex(embedding_dim, 'angular')  # 使用余弦相似度
    annoy_index.set_seed(2020)
    

    #添加所有物品的embedding索引
    for idx, item_emb in enumerate(tqdm(all_item_embeddings_cpu, desc="添加到物品索引")):
        annoy_index.add_item(idx, item_emb)

    annoy_index.build(200)
    
    
    # 3. 准备所有有效用户的数据
    valid_queries = []
    valid_user_data = []
    
    log.info("准备用户数据...")
    for idx, (user_id, item_id) in enumerate(tqdm(df_query.values, desc="准备数据")):
        if user_id not in user_item_dict:
            continue
            
        try:
            # 编码用户ID
            user_id_encoded = user_encoder.transform([user_id])[0] + 1
            
            # 构建历史序列
            hist_items = user_item_dict[user_id]

            #编码用户交互过的商品id
            hist_encoded = item_encoder.transform(hist_items) + 1
            
            # Padding处理
            if len(hist_encoded) > max_seq_len:
                hist_encoded = hist_encoded[-max_seq_len:]
            else:
                hist_encoded = np.pad(hist_encoded, (max_seq_len - len(hist_encoded), 0), 
                                    mode='constant', constant_values=0)
            
            valid_queries.append((user_id, item_id, hist_items))
            valid_user_data.append((user_id_encoded, hist_encoded))
            
        except Exception as e:
            log.warning(f"用户 {user_id} 数据准备失败: {e}")
            continue
    
    log.info(f"准备处理 {len(valid_user_data)} 个用户")

    
    if len(valid_user_data) == 0:
        log.warning("没有有效用户数据!")
        return pd.DataFrame()
    

    # 3. 批量处理用户embedding
    all_user_embeddings = []
    
    for i in tqdm(range(0, len(valid_user_data), batch_size), desc="批量计算用户embedding"):
        batch_end = min(i + batch_size, len(valid_user_data))
        batch_data = valid_user_data[i:batch_end]
        
        # 准备批数据
        batch_user_ids = []
        batch_histories = []
        
        for user_encoded, hist_encoded in batch_data:
            batch_user_ids.append(user_encoded)
            batch_histories.append(hist_encoded)
        
        # 转换为GPU张量
        batch_user_tensor = torch.LongTensor(batch_user_ids).to(device)
        batch_hist_tensor = torch.LongTensor(batch_histories).to(device)
        
        # 批量推理
        with torch.no_grad():
            batch_user_emb = model.get_user_embeddings(batch_user_tensor, batch_hist_tensor)
            # 转换到CPU
            batch_user_emb_cpu = batch_user_emb.cpu().numpy()
            all_user_embeddings.append(batch_user_emb_cpu)
    
    # 拼接所有用户embedding
    all_user_embeddings = np.vstack(all_user_embeddings)  # [num_users, embedding_dim]
    log.info(f"用户embedding形状: {all_user_embeddings.shape}")

    
    # 使用AnnoyIndex进行相似度查询
    all_topk_items = []
    all_topk_scores = []

    for user_idx in tqdm(range(len(all_user_embeddings)), desc="AnnoyIndex查询"):
        user_emb = all_user_embeddings[user_idx]
        
        # 查询top-k最相似的物品
        item_indices, distances = annoy_index.get_nns_by_vector(
            user_emb, top_k, include_distances=True
        )
        
        # 转换距离为相似度分数（距离越小，相似度越大）
        # 对于angular距离，相似度 = 2 - distance
        similarity_scores = [2 - distance for distance in distances]
        
        all_topk_items.append(item_indices)
        all_topk_scores.append(similarity_scores)
    
    log.info("构建最终召回结果...")

    
    # 6. 构建最终结果
    for user_idx, (user_id, item_id, hist_items) in enumerate(tqdm(valid_queries, desc="构建结果")):
        try:
            # 获取该用户的top-k推荐
            user_topk_indices = all_topk_items[user_idx]
            user_topk_scores = all_topk_scores[user_idx]
            
            # 解码物品ID（AnnoyIndex返回的是索引，需要映射回原始物品ID）
            recalled_items_encoded = np.array(user_topk_indices) + 1  # AnnoyIndex索引 + 1 = 编码ID
            recalled_items = item_encoder.inverse_transform(recalled_items_encoded - 1)
            
            # 过滤已交互物品
            filtered_items = []
            filtered_scores = []
            
            for item, score in zip(recalled_items, user_topk_scores):
                if item not in hist_items:
                    filtered_items.append(item)
                    filtered_scores.append(score)
                if len(filtered_items) >= recall_num:
                    break
            
            if len(filtered_items) == 0:
                continue
                
            # 构建结果DataFrame
            df_temp = pd.DataFrame()
            df_temp['article_id'] = filtered_items
            df_temp['sim_score'] = filtered_scores
            df_temp['user_id'] = user_id
            
            if item_id == -1:
                df_temp['label'] = np.nan  # 测试集
            else:
                df_temp['label'] = 0
                df_temp.loc[df_temp['article_id'] == item_id, 'label'] = 1
            
            df_temp = df_temp[['user_id', 'article_id', 'sim_score', 'label']]
            df_temp['user_id'] = df_temp['user_id'].astype('int')
            df_temp['article_id'] = df_temp['article_id'].astype('int')
            
            all_results.append(df_temp)
            
        except Exception as e:
            log.warning(f"处理用户 {user_id} 结果时出错: {e}")
            continue
    
    if len(all_results) == 0:
        log.warning("没有生成任何召回结果!")
        return pd.DataFrame()
    
    # 合并所有结果
    df_results = pd.concat(all_results, ignore_index=True)
    
    # 排序
    df_results = df_results.sort_values(['user_id', 'sim_score'], 
                                       ascending=[True, False]).reset_index(drop=True)
    
    log.info(f"召回完成! 总结果数: {len(df_results)}")
    log.info(f"覆盖用户数: {df_results['user_id'].nunique()}")
    
    return df_results


def main():
    # 数据路径设置
    if args.mode == 'valid':
        data_path = '/home/wangtiantian/shikainan/newscommemder/top2wk/user_data/data/offline'
        model_dir = '/home/wangtiantian/shikainan/newscommemder/top2wk/user_data/model/offline'
        output_path = '/home/wangtiantian/shikainan/newscommemder/top2wk/user_data/data/offline/recall_youtube_dnn.pkl'
    else:
        data_path = '/home/wangtiantian/shikainan/newscommemder/top2wk/user_data/data/online'
        model_dir = '/home/wangtiantian/shikainan/newscommemder/top2wk/user_data/model/online'
        output_path = '/home/wangtiantian/shikainan/newscommemder/top2wk/user_data/data/online/recall_youtube_dnn.pkl'
    
    # 加载模型
    try:
        model, user_encoder, item_encoder, model_config = load_model_and_encoders(model_dir)
    except FileNotFoundError as e:
        log.error(f"模型加载失败: {e}")
        log.error("请先运行 train_youtube_dnn.py 进行模型训练!")
        return
    
    # 读取数据
    click_path = os.path.join(data_path, 'click.pkl')
    query_path = os.path.join(data_path, 'query.pkl')
    
    if not os.path.exists(click_path) or not os.path.exists(query_path):
        log.error(f"数据文件不存在: {click_path} 或 {query_path}")
        return
    
    df_click = pd.read_pickle(click_path)
    df_query = pd.read_pickle(query_path)
    
    log.info(f'点击数据形状: {df_click.shape}')
    log.info(f'查询数据形状: {df_query.shape}')
    
    # 构建用户-物品交互映射
    user_item_dict = build_user_item_mappings(df_click)
    log.info(f"用户交互字典构建完成，用户数: {len(user_item_dict)}")
    
    # 批量召回
    df_results = batch_recall_optimized(
        df_query, model, user_encoder, item_encoder, user_item_dict, 
        model_config, args.batch_size, args.top_k, args.recall_num
    )
    
    if len(df_results) == 0:
        log.error("召回结果为空!")
        return
    
    log.debug(f'召回结果示例:\n{df_results.head()}')
    
    # 计算召回指标（仅验证模式）
    if args.mode == 'valid':
        log.info("计算召回指标...")
        valid_queries = df_query[df_query['click_article_id'] != -1]
        total_users = valid_queries['user_id'].nunique()
        
        if total_users > 0:
            valid_results = df_results[df_results['label'].notnull()]
            
            if len(valid_results) > 0:
                hitrate_5, mrr_5, hitrate_10, mrr_10, hitrate_20, mrr_20, \
                hitrate_40, mrr_40, hitrate_50, mrr_50 = evaluate(valid_results, total_users)
                
                log.info("YouTube DNN召回指标:")
                log.info(f"  Hit@5: {hitrate_5:.4f}, MRR@5: {mrr_5:.4f}")
                log.info(f"  Hit@10: {hitrate_10:.4f}, MRR@10: {mrr_10:.4f}")
                log.info(f"  Hit@20: {hitrate_20:.4f}, MRR@20: {mrr_20:.4f}")
                log.info(f"  Hit@40: {hitrate_40:.4f}, MRR@40: {mrr_40:.4f}")
                log.info(f"  Hit@50: {hitrate_50:.4f}, MRR@50: {mrr_50:.4f}")
            else:
                log.warning("没有有效的验证结果用于指标计算!")
        else:
            log.warning("没有有效的验证查询!")
    
    # 保存结果
    df_results.to_pickle(output_path)
    log.info(f"召回结果保存到: {output_path}")
    log.info("YouTube DNN召回完成!")


if __name__ == '__main__':
    main()