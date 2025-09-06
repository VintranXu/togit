import argparse
import os
import pickle
import random
import signal
import warnings
from collections import defaultdict
from itertools import permutations
from random import shuffle

# 基础配置和导入
import multitasking
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import Logger, evaluate

warnings.filterwarnings('ignore')

#多任务并行配置
max_threads = multitasking.config['CPU_CORES']
multitasking.set_max_threads(max_threads)
multitasking.set_engine('process')
signal.signal(signal.SIGINT, multitasking.killall)

random.seed(2020)

# 命令行参数
parser = argparse.ArgumentParser(description='召回合并')
parser.add_argument('--mode', default='valid')
parser.add_argument('--logfile', default='test.log')

args = parser.parse_args()

mode = args.mode
logfile = args.logfile

# 初始化日志
os.makedirs('/home/wangtiantian/shikainan/newscommemder/top2wk/user_data/log', exist_ok=True)
log = Logger(f'/home/wangtiantian/shikainan/newscommemder/top2wk/user_data/log/{logfile}').logger
log.info(f'召回合并: {mode}')


#对每个用户返回的分数进行归一化
def mms(df):
    user_score_max = {}
    user_score_min = {}

    # 获取用户下的相似度的最大值和最小值
    # for user_id, g in df[['user_id', 'sim_score']].groupby('user_id'):
    for user_id, g in df.groupby('user_id')[['user_id', 'sim_score']]:
        
        scores = g['sim_score'].values.tolist()
        user_score_max[user_id] = scores[0]
        user_score_min[user_id] = scores[-1]

    ans = []
    for user_id, sim_score in tqdm(df[['user_id', 'sim_score']].values):
        ans.append((sim_score - user_score_min[user_id]) /
                   (user_score_max[user_id] - user_score_min[user_id]) +
                   10**-3)
    return ans


#计算两种不同召回策略之间的重合度，有多少内容重合了
def recall_result_sim(df1_, df2_):
    df1 = df1_.copy()
    df2 = df2_.copy()

    user_item_ = df1.groupby('user_id')['article_id'].agg(set).reset_index()
    user_item_dict1 = dict(zip(user_item_['user_id'],
                               user_item_['article_id']))

    user_item_ = df2.groupby('user_id')['article_id'].agg(set).reset_index()
    user_item_dict2 = dict(zip(user_item_['user_id'],
                               user_item_['article_id']))

    cnt = 0
    hit_cnt = 0

    for user in user_item_dict1.keys():
        item_set1 = user_item_dict1[user]

        cnt += len(item_set1)

        if user in user_item_dict2:
            item_set2 = user_item_dict2[user]

            inters = item_set1 & item_set2
            hit_cnt += len(inters)

    return hit_cnt / cnt


if __name__ == '__main__':
    if mode == 'valid':
        df_click = pd.read_pickle('/home/wangtiantian/shikainan/newscommemder/top2wk/user_data/data/offline/click.pkl')
        df_query = pd.read_pickle('/home/wangtiantian/shikainan/newscommemder/top2wk/user_data/data/offline/query.pkl')

        recall_path = '/home/wangtiantian/shikainan/newscommemder/top2wk/user_data/data/offline'
    else:
        df_click = pd.read_pickle('/home/wangtiantian/shikainan/newscommemder/top2wk/user_data/data/online/click.pkl')
        df_query = pd.read_pickle('/home/wangtiantian/shikainan/newscommemder/top2wk/user_data/data/online/query.pkl')

        recall_path = '/home/wangtiantian/shikainan/newscommemder/top2wk/user_data/data/online'

    log.debug(f'max_threads {max_threads}')

    recall_methods = ['itemcf', 'w2v', 'youtube_dnn']

    weights = {'itemcf': 1, 'youtube_dnn': 0.5, 'w2v': 0.1}

    # 加载、归一化并加权各路召回结果
    recall_list = []
    recall_dict = {}
    for recall_method in recall_methods:
        recall_result = pd.read_pickle(
            f'{recall_path}/recall_{recall_method}.pkl')
        

        weight = weights[recall_method]

        recall_result['sim_score'] = mms(recall_result)
        recall_result['sim_score'] = recall_result['sim_score'] * weight

        recall_list.append(recall_result)
        recall_dict[recall_method] = recall_result

    # 求相似度,# for循环，遍历所有召回策略的两两组合
    for recall_method1, recall_method2 in permutations(recall_methods, 2):
        score = recall_result_sim(recall_dict[recall_method1],
                                  recall_dict[recall_method2])
        log.debug(f'召回相似度 {recall_method1}-{recall_method2}: {score}')

    # 合并召回结果
    recall_final = pd.concat(recall_list, sort=False)

    # 专门计算每个 (user_id, article_id) 组合的最终融合分数。
    recall_score = recall_final[['user_id', 'article_id',
                                 'sim_score']].groupby([
                                     'user_id', 'article_id'
                                 ])['sim_score'].sum().reset_index()
    
    # 创建唯一的 "用户-文章" 基础表
    recall_final = recall_final[['user_id', 'article_id', 'label'
                                 ]].drop_duplicates(['user_id', 'article_id'])
    recall_final = recall_final.merge(recall_score, how='left')

    recall_final.sort_values(['user_id', 'sim_score'],
                             inplace=True,
                             ascending=[True, False])

    log.debug(f'recall_final.shape: {recall_final.shape}')
    log.debug(f'recall_final: {recall_final.head()}')

    # 删除无正样本的训练集用户
    gg = recall_final.groupby(['user_id'])
    useful_recall = []

    for user_id, g in tqdm(gg):
        #判断是否是测试用户
        if g['label'].isnull().sum() > 0:
            useful_recall.append(g)
        else:
            #说明是验证用户
            label_sum = g['label'].sum()
            if label_sum > 1:
                print('error', user_id)
            elif label_sum == 1:
                #只保留召回了正样本的用户数据
                useful_recall.append(g)
    # 这是为后续排序模型准备高质量训练数据的关键一步。排序模型需要学习区分正样本（用户点击了）和负样本（没点击）。

    # 将筛选出的“有用”数据重新拼接成一个DataFrame
    df_useful_recall = pd.concat(useful_recall, sort=False)
    log.debug(f'df_useful_recall: {df_useful_recall.head()}')
    # 这个操作会创建一个新的、从0开始的连续行号，使DataFrame更规整。
    df_useful_recall = df_useful_recall.sort_values(
        ['user_id', 'sim_score'], ascending=[True,
                                             False]).reset_index(drop=True)

    # 计算相关指标
    if mode == 'valid':
        #说明是验证集中的内容
        total = df_query[df_query['click_article_id'] != -1].user_id.nunique()
        hitrate_5, mrr_5, hitrate_10, mrr_10, hitrate_20, mrr_20, hitrate_40, mrr_40, hitrate_50, mrr_50 = evaluate(
            df_useful_recall[df_useful_recall['label'].notnull()], total)
        hitrate_5, mrr_5, hitrate_10, mrr_10, hitrate_20, mrr_20, hitrate_40, mrr_40, hitrate_50, mrr_50

        log.debug(
            f'召回合并后指标: {hitrate_5}, {mrr_5}, {hitrate_10}, {mrr_10}, {hitrate_20}, {mrr_20}, {hitrate_40}, {mrr_40}, {hitrate_50}, {mrr_50}'
        )

    df = df_useful_recall['user_id'].value_counts().reset_index()
    df.columns = ['user_id', 'cnt']
    log.debug(f"平均每个用户召回数量：{df['cnt'].mean()}")

    log.debug(
        f"标签分布: {df_useful_recall[df_useful_recall['label'].notnull()]['label'].value_counts()}"
    )

    # 保存到本地
    if mode == 'valid':
        df_useful_recall.to_pickle('/home/wangtiantian/shikainan/newscommemder/top2wk/user_data/data/offline/recall.pkl')
    else:
        df_useful_recall.to_pickle('/home/wangtiantian/shikainan/newscommemder/top2wk/user_data/data/online/recall.pkl')
