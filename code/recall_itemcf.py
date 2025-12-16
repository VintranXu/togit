import argparse
import math
import os
import pickle
import random
import signal
from collections import defaultdict
from random import shuffle

import multitasking
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import Logger, evaluate

#设置多线程和日志
max_threads = multitasking.config['CPU_CORES']
multitasking.set_max_threads(max_threads)
multitasking.set_engine('process')
signal.signal(signal.SIGINT, multitasking.killall)

random.seed(2020)

# 命令行参数
parser = argparse.ArgumentParser(description='itemcf 召回')
parser.add_argument('--mode', default='valid')
parser.add_argument('--logfile', default='test.log')

args = parser.parse_args()

mode = args.mode
logfile = args.logfile

# 初始化日志
os.makedirs('./user_data/log', exist_ok=True)
log = Logger(f'./user_data/log/{logfile}').logger
log.info(f'itemcf 召回，mode: {mode}')


#计算物品相似度
def cal_sim(df):
    """
    基于用户点击行为计算物品之间的相似度（ItemCF协同过滤算法）
    
    参数:
        df: DataFrame, 包含 user_id 和 click_article_id 列
    
    返回:
        sim_dict: 物品相似度字典，格式为 {item_i: {item_j: similarity_ij}}
        user_item_dict: 用户点击记录字典，格式为 {user_id: [item_list]}
    """
    
    # ========== 步骤1: 构建用户-物品点击字典 ==========
    # 将每个用户的点击记录聚合成列表
    # 例如: user_1 -> [item_a, item_b, item_c]
    user_item_ = df.groupby('user_id')['click_article_id'].agg(
        lambda x: list(x)).reset_index()
    user_item_dict = dict(
        zip(user_item_['user_id'], user_item_['click_article_id']))

    # ========== 步骤2: 初始化统计变量 ==========
    # item_cnt: 记录每个物品的总点击次数（用于后续归一化）
    # sim_dict: 存储物品之间的相似度分数
    item_cnt = defaultdict(int)
    sim_dict = {}

    # ========== 步骤3: 遍历用户点击记录，计算物品共现关系 ==========
    # 核心思想: 如果两个物品被同一个用户点击过，说明它们有相似性
    for _, items in tqdm(user_item_dict.items()):
        # items: 当前用户点击的物品列表，例如 [item1, item2, item3, item4]
        
        for loc1, item in enumerate(items):
            # loc1: 当前物品在点击序列中的位置索引
            # item: 当前物品ID
            
            # 统计物品的总点击次数
            item_cnt[item] += 1
            
            # 为当前物品初始化相似度字典（如果不存在）
            sim_dict.setdefault(item, {})

            # 遍历同一用户点击的其他物品，计算共现关系
            for loc2, relate_item in enumerate(items):
                # loc2: 相关物品在点击序列中的位置索引
                # relate_item: 相关物品ID
                
                # 跳过物品自己与自己的相似度计算
                if item == relate_item:
                    continue

                # 为物品对初始化相似度分数（如果不存在）
                sim_dict[item].setdefault(relate_item, 0)

                # ========== 位置信息权重计算 ==========
                # 考虑文章的点击顺序对相似度的影响
                
                # loc_alpha: 正向/反向点击权重因子
                # - 正向点击 (loc2 > loc1): 权重为 1.0，表示先点击item后点击relate_item
                # - 反向点击 (loc2 < loc1): 权重为 0.7，表示先点击relate_item后点击item
                # 正向点击权重更高，因为用户的点击顺序反映了兴趣转移
                loc_alpha = 1.0 if loc2 > loc1 else 0.7
                
                # loc_weight: 时间衰减权重
                # - 使用指数衰减: 0.9^(距离-1)
                # - 距离越远，权重越低
                # - 例如: 相邻物品距离为1，权重为 loc_alpha * 0.9^0 = loc_alpha
                #        距离为2的物品，权重为 loc_alpha * 0.9^1 = 0.9 * loc_alpha
                loc_weight = loc_alpha * (0.9**(np.abs(loc2 - loc1) - 1))

                # ========== 累加相似度分数 ==========
                # 分母: math.log(1 + len(items)) 
                # - 对点击数量多的用户进行惩罚，避免活跃用户主导相似度
                # - 例如: 点击100个物品的用户贡献的权重比点击5个物品的用户低
                sim_dict[item][relate_item] += loc_weight  / \
                    math.log(1 + len(items))

    # ========== 步骤4: 归一化相似度 ==========
    # 使用余弦相似度的思想进行归一化
    # 公式: sim(i,j) = c_ij / sqrt(N(i) * N(j))
    # 其中:
    #   - c_ij: 物品i和j的共现权重累加值
    #   - N(i): 物品i的总点击次数
    #   - N(j): 物品j的总点击次数
    # 这样可以消除热门物品的影响，使相似度更加准确
    for item, relate_items in tqdm(sim_dict.items()):
        for relate_item, cij in relate_items.items():
            sim_dict[item][relate_item] = cij / \
                math.sqrt(item_cnt[item] * item_cnt[relate_item])

    return sim_dict, user_item_dict


@multitasking.task
def recall(df_query, item_sim, user_item_dict, worker_id):
    """
    基于物品协同过滤(ItemCF)的召回函数，为每个用户生成推荐文章列表

    参数:
        df_query: DataFrame, 包含以下列:
            - user_id: 用户ID
            - click_article_id: 用户点击的文章ID
        item_sim: dict, 物品相似度字典，格式为 {item_i: {item_j: similarity_ij}}
        user_item_dict: dict, 用户点击记录字典，格式为 {user_id: [item_list]}
        worker_id: int, 工作线程ID，用于保存临时文件的命名

    返回:
        无返回值，结果保存到 ./user_data/tmp/itemcf/{worker_id}.pkl 文件中
    """
    # 存储所有用户的推荐结果
    data_list = []

    # ========== 步骤1: 遍历每个用户，生成推荐列表 ==========
    for user_id, item_id in tqdm(df_query.values):
        # rank: 候选物品的累计相似度分数字典
        rank = {}

        # ========== 步骤2: 过滤无历史记录的用户 ==========
        # 如果用户从未点击过任何物品，无法进行协同过滤推荐，跳过该用户
        if user_id not in user_item_dict:
            continue

        # ========== 步骤3: 获取用户最近的交互历史 ==========
        interacted_items = user_item_dict[user_id]
        # 将用户历史逆序（[::-1]），优先使用最近点击的物品
        # 只保留最近的2个物品（[:2]），减少计算量并聚焦于近期兴趣
        # 例如: [item1, item2, item3, item4, item5] -> [item5, item4]
        interacted_items = interacted_items[::-1][:2]

        # ========== 步骤4: 基于用户历史物品查找相似物品 ==========
        # 对于用户点击过的每个物品，找到与其最相似的物品作为推荐候选
        for loc, item in enumerate(interacted_items):
            # loc: 物品在历史列表中的位置（0表示最近点击，1表示次近点击）
            # item: 当前历史物品ID
            
            # 获取与当前物品最相似的Top 200个物品
            # sorted(..., reverse=True): 按相似度降序排序
            # [0:200]: 只取前200个最相似的物品
            for relate_item, wij in sorted(item_sim[item].items(),
                                           key=lambda d: d[1],
                                           reverse=True)[0:200]:
                # relate_item: 相似物品ID
                # wij: 相似度分数
                
                # 排除用户已经点击过的物品（避免重复推荐）
                if relate_item not in interacted_items:
                    # 初始化候选物品的分数
                    rank.setdefault(relate_item, 0)
                    
                    # ========== 分数累加与时间衰减 ==========
                    # wij: 物品相似度
                    # 0.7**loc: 时间衰减因子
                    #   - loc=0 (最近点击): 权重为 0.7^0 = 1.0
                    #   - loc=1 (次近点击): 权重为 0.7^1 = 0.7
                    # 越近期的点击对推荐的影响越大
                    rank[relate_item] += wij * (0.7**loc)


        # ========== 步骤5: 排序并截取Top N推荐结果 ==========
        # 按累计相似度分数降序排序，截取Top 100作为最终召回结果
        sim_items = sorted(rank.items(), key=lambda d: d[1],
                           reverse=True)[:100]
        
        # 分离物品ID和相似度分数到不同列表
        item_ids = [item[0] for item in sim_items]
        item_sim_scores = [item[1] for item in sim_items]

        # ========== 步骤6: 构建标准格式的推荐结果DataFrame ==========
        df_temp = pd.DataFrame()
        df_temp['article_id'] = item_ids          # 推荐的文章ID列表
        df_temp['sim_score'] = item_sim_scores    # 对应的相似度分数
        df_temp['user_id'] = user_id              # 用户ID

        # ========== 步骤7: 添加标签列（用于训练/验证） ==========
        if item_id == -1:
            # 测试集场景: item_id=-1 表示没有真实标签，设为 NaN
            df_temp['label'] = np.nan
        else:
            # 训练/验证集场景: 需要标注正负样本
            # 默认所有推荐物品都是负样本（label=0）
            df_temp['label'] = 0
            # 如果推荐列表中包含用户真实点击的文章，标记为正样本（label=1）
            df_temp.loc[df_temp['article_id'] == item_id, 'label'] = 1

        # ========== 步骤8: 数据格式化与类型转换 ==========
        # 重新排列列顺序，确保格式统一
        df_temp = df_temp[['user_id', 'article_id', 'sim_score', 'label']]
        # 强制转换ID为整数类型（节省内存，提升后续处理效率）
        df_temp['user_id'] = df_temp['user_id'].astype('int')
        df_temp['article_id'] = df_temp['article_id'].astype('int')

        # 将当前用户的推荐结果加入结果列表
        data_list.append(df_temp)

    # ========== 步骤9: 合并所有用户的推荐结果 ==========
    df_data = pd.concat(data_list, sort=False)

    # ========== 步骤10: 保存当前工作线程的召回结果 ==========
    # 确保目标目录存在
    os.makedirs('./user_data/tmp/itemcf', exist_ok=True)

    # 使用 worker_id 命名文件，支持多线程并行保存
    # 例如: ./user_data/tmp/itemcf/0.pkl, ./user_data/tmp/itemcf/1000.pkl, ...
    df_data.to_pickle(f'./user_data/tmp/itemcf/{worker_id}.pkl')


if __name__ == '__main__':
    if mode == 'valid':
        df_click = pd.read_pickle('./user_data/data/offline/click.pkl')
        df_query = pd.read_pickle('./user_data/data/offline/query.pkl')

        os.makedirs('./user_data/sim/offline', exist_ok=True)
        sim_pkl_file = './user_data/sim/offline/itemcf_sim.pkl'
    else:
        df_click = pd.read_pickle('./user_data/data/online/click.pkl')
        df_query = pd.read_pickle('./user_data/data/online/query.pkl')

        os.makedirs('./user_data/sim/online', exist_ok=True)
        sim_pkl_file = './user_data/sim/online/itemcf_sim.pkl'

    log.debug(f'df_click shape: {df_click.shape}')
    log.debug(f'{df_click.head()}')

    #计算相似度，并保存
    item_sim, user_item_dict = cal_sim(df_click)
    f = open(sim_pkl_file, 'wb')
    pickle.dump(item_sim, f)

    f.close()

    # 召回
    n_split = max_threads
    # 获取要召回的用户的id
    all_users = df_query['user_id'].unique()  # df_query.shape：(90258, 2) all_users.shape：(90258,)
        # df_click.shape
        # (1590375, 9)
        # df_click['user_id'].unique().shape
        # (250000,)
    # 打乱用户id
    shuffle(all_users)
    # 每个线程要处理的用户数
    total = len(all_users)


    n_len = total // n_split

    # 清空临时文件夹
    for path, _, file_list in os.walk('./user_data/tmp/itemcf'):
        for file_name in file_list:
            os.remove(os.path.join(path, file_name))

    for i in range(0, total, n_len):
        part_users = all_users[i:i + n_len]
        df_temp = df_query[df_query['user_id'].isin(part_users)]
        recall(df_temp, item_sim, user_item_dict, i)

    multitasking.wait_for_tasks()
    log.info('合并任务')


    df_data = pd.DataFrame()
    for path, _, file_list in os.walk('./user_data/tmp/itemcf'):
        for file_name in file_list:
            df_temp = pd.read_pickle(os.path.join(path, file_name))
            df_data = pd.concat([df_data, df_temp], ignore_index=True)  # 注意是列表形式

    # ========== 排序召回结果 ==========
    # 必须进行排序，原因：
    # 1. 多线程并行处理导致数据无序
    # 2. 后续评估和模型训练需要有序数据
    # 排序规则：
    # - 第一优先级：按 user_id 升序 (ascending=True)，将同一用户的推荐结果聚集在一起
    # - 第二优先级：按 sim_score 降序 (ascending=False)，每个用户的推荐按相似度从高到低排列
    # reset_index(drop=True)：重置行索引为连续的整数序列，drop=True 表示丢弃原索引列
    df_data = df_data.sort_values(['user_id', 'sim_score'],
                                  ascending=[True,
                                             False]).reset_index(drop=True)
    log.debug(f'df_data.head: {df_data.head()}')

    # 计算召回指标
    if mode == 'valid':
        log.info(f'计算召回指标')

        total = df_query[df_query['click_article_id'] != -1].user_id.nunique()

        hitrate_5, mrr_5, hitrate_10, mrr_10, hitrate_20, mrr_20, hitrate_40, mrr_40, hitrate_50, mrr_50 = evaluate(
            df_data[df_data['label'].notnull()], total)

        log.debug(
            f'itemcf: {hitrate_5}, {mrr_5}, {hitrate_10}, {mrr_10}, {hitrate_20}, {mrr_20}, {hitrate_40}, {mrr_40}, {hitrate_50}, {mrr_50}'
        )
    # 保存召回结果
    if mode == 'valid':
        df_data.to_pickle('./user_data/data/offline/recall_itemcf.pkl')
    else:
        df_data.to_pickle('./user_data/data/online/recall_itemcf.pkl')
