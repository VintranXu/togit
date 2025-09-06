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
os.makedirs('/home/wangtiantian/shikainan/newscommemder/top2wk/user_data/log', exist_ok=True)
log = Logger(f'/home/wangtiantian/shikainan/newscommemder/top2wk/user_data/log/{logfile}').logger
log.info(f'itemcf 召回，mode: {mode}')


#计算物品相似度
def cal_sim(df):
    # 创建一个字典，存储每个用户与其点击的文章列表
    user_item_ = df.groupby('user_id')['click_article_id'].agg(
        lambda x: list(x)).reset_index()
    user_item_dict = dict(
        zip(user_item_['user_id'], user_item_['click_article_id']))


    item_cnt = defaultdict(int)
    sim_dict = {}

    for _, items in tqdm(user_item_dict.items()):
        for loc1, item in enumerate(items):
            item_cnt[item] += 1
            sim_dict.setdefault(item, {})

            for loc2, relate_item in enumerate(items):
                if item == relate_item:
                    continue

                sim_dict[item].setdefault(relate_item, 0)

                # 位置信息权重
                # 考虑文章的正向顺序点击和反向顺序点击
                loc_alpha = 1.0 if loc2 > loc1 else 0.7
                loc_weight = loc_alpha * (0.9**(np.abs(loc2 - loc1) - 1))

                sim_dict[item][relate_item] += loc_weight  / \
                    math.log(1 + len(items))

    for item, relate_items in tqdm(sim_dict.items()):
        for relate_item, cij in relate_items.items():
            sim_dict[item][relate_item] = cij / \
                math.sqrt(item_cnt[item] * item_cnt[relate_item])

    return sim_dict, user_item_dict


@multitasking.task
def recall(df_query, item_sim, user_item_dict, worker_id):
    data_list = []

    for user_id, item_id in tqdm(df_query.values):
        rank = {}

        #没有交互过，即没有交互过任何物品，则跳过此用户的推荐过程。
        if user_id not in user_item_dict:
            continue

        interacted_items = user_item_dict[user_id]
        # 将用户历史逆序，优先使用最近点击的两个物品
        interacted_items = interacted_items[::-1][:2]

        #取出每个物品top200最相似的物品
        for loc, item in enumerate(interacted_items):
            for relate_item, wij in sorted(item_sim[item].items(),
                                           key=lambda d: d[1],
                                           reverse=True)[0:200]:
                if relate_item not in interacted_items:
                    rank.setdefault(relate_item, 0)
                    # 分数累加与时间衰减
                    rank[relate_item] += wij * (0.7**loc)


        # 按累计相似度分数降序排序
        # 截取Top 100作为最终召回结果
        # 分离物品ID和分数到不同列表
        sim_items = sorted(rank.items(), key=lambda d: d[1],
                           reverse=True)[:100]
        item_ids = [item[0] for item in sim_items]
        item_sim_scores = [item[1] for item in sim_items]

        # 构建标准格式的推荐结果DataFrame。
        df_temp = pd.DataFrame()
        df_temp['article_id'] = item_ids
        df_temp['sim_score'] = item_sim_scores
        df_temp['user_id'] = user_id

        
        if item_id == -1:
            df_temp['label'] = np.nan  # 测试集，无真实标签
        else:
            #如果只是推荐的文档但不是真实文章，则为负样本，如果是真实文章则为正样本
            df_temp['label'] = 0
            df_temp.loc[df_temp['article_id'] == item_id, 'label'] = 1

        # 重新排列列顺序
        # 强制转换ID为整数类型（节省内存）
        # 加入结果列表
        df_temp = df_temp[['user_id', 'article_id', 'sim_score', 'label']]
        df_temp['user_id'] = df_temp['user_id'].astype('int')
        df_temp['article_id'] = df_temp['article_id'].astype('int')

        data_list.append(df_temp)

    df_data = pd.concat(data_list, sort=False)

    os.makedirs('/home/wangtiantian/shikainan/newscommemder/top2wk/user_data/tmp/itemcf', exist_ok=True)

    #并行存储召回结果
    df_data.to_pickle(f'/home/wangtiantian/shikainan/newscommemder/top2wk/user_data/tmp/itemcf/{worker_id}.pkl')


if __name__ == '__main__':
    if mode == 'valid':
        df_click = pd.read_pickle('/home/wangtiantian/shikainan/newscommemder/top2wk/user_data/data/offline/click.pkl')
        df_query = pd.read_pickle('/home/wangtiantian/shikainan/newscommemder/top2wk/user_data/data/offline/query.pkl')

        os.makedirs('/home/wangtiantian/shikainan/newscommemder/top2wk/user_data/sim/offline', exist_ok=True)
        sim_pkl_file = '/home/wangtiantian/shikainan/newscommemder/top2wk/user_data/sim/offline/itemcf_sim.pkl'
    else:
        df_click = pd.read_pickle('/home/wangtiantian/shikainan/newscommemder/top2wk/user_data/data/online/click.pkl')
        df_query = pd.read_pickle('/home/wangtiantian/shikainan/newscommemder/top2wk/user_data/data/online/query.pkl')

        os.makedirs('/home/wangtiantian/shikainan/newscommemder/top2wk/user_data/sim/online', exist_ok=True)
        sim_pkl_file = '/home/wangtiantian/shikainan/newscommemder/top2wk/user_data/sim/online/itemcf_sim.pkl'

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
    all_users = df_query['user_id'].unique()
    # 打乱用户id
    shuffle(all_users)
    # 每个线程要处理的用户数
    total = len(all_users)


    n_len = total // n_split

    # 清空临时文件夹
    for path, _, file_list in os.walk('/home/wangtiantian/shikainan/newscommemder/top2wk/user_data/tmp/itemcf'):
        for file_name in file_list:
            os.remove(os.path.join(path, file_name))

    for i in range(0, total, n_len):
        part_users = all_users[i:i + n_len]
        df_temp = df_query[df_query['user_id'].isin(part_users)]
        recall(df_temp, item_sim, user_item_dict, i)

    multitasking.wait_for_tasks()
    log.info('合并任务')


    df_data = pd.DataFrame()
    for path, _, file_list in os.walk('/home/wangtiantian/shikainan/newscommemder/top2wk/user_data/tmp/itemcf'):
        for file_name in file_list:
            df_temp = pd.read_pickle(os.path.join(path, file_name))
            df_data = pd.concat([df_data, df_temp], ignore_index=True)  # 注意是列表形式

    # 必须加，对其进行排序
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
        df_data.to_pickle('/home/wangtiantian/shikainan/newscommemder/top2wk/user_data/data/offline/recall_itemcf.pkl')
    else:
        df_data.to_pickle('/home/wangtiantian/shikainan/newscommemder/top2wk/user_data/data/online/recall_itemcf.pkl')
