import argparse
import math
import os
import pickle
import random
import signal
import warnings
from collections import defaultdict
from random import shuffle

import multitasking
import numpy as np
import pandas as pd
from annoy import AnnoyIndex
from gensim.models import Word2Vec
from tqdm import tqdm

from utils import Logger, evaluate

warnings.filterwarnings('ignore')

#指定多线程的任务
max_threads = multitasking.config['CPU_CORES']
multitasking.set_max_threads(max_threads)
multitasking.set_engine('process')
signal.signal(signal.SIGINT, multitasking.killall)

seed = 2020
random.seed(seed)

# 命令行参数
parser = argparse.ArgumentParser(description='w2v 召回')
parser.add_argument('--mode', default='valid')
parser.add_argument('--logfile', default='test.log')

args = parser.parse_args()

mode = args.mode
logfile = args.logfile

# 初始化日志
os.makedirs('/home/wangtiantian/shikainan/newscommemder/top2wk/user_data/log', exist_ok=True)
log = Logger(f'/home/wangtiantian/shikainan/newscommemder/top2wk/user_data/log/{logfile}').logger
log.info(f'w2v 召回，mode: {mode}')


#进行word2vec的训练
def word2vec(df_, f1, f2, model_path):
    df = df_.copy()

    #按用户分组，将每个用户点击的文章聚合成列表
    tmp = df.groupby(f1, as_index=False)[f2].agg(
        {'{}_{}_list'.format(f1, f2): list})
    
    #转化为列表形式
    sentences = tmp['{}_{}_list'.format(f1, f2)].values.tolist()
    del tmp['{}_{}_list'.format(f1, f2)]

    # 再将每个用户和文章的交互的列表转化为字符串形式 
    words = []
    for i in range(len(sentences)):
        x = [str(x) for x in sentences[i]]
        sentences[i] = x
        words += x

    #开启word2vec的训练
    if os.path.exists(f'{model_path}/w2v.m'):
        model = Word2Vec.load(f'{model_path}/w2v.m')
    else:
        #使用的是skipgram
        #由sg=1这个参数决定
        model = Word2Vec(sentences=sentences,
                         vector_size =256,
                         window=3,
                         min_count=1,
                         sg=1,
                         hs=0,
                         seed=seed,
                         negative=5,
                         workers=10,
                         epochs=1)
        model.save(f'{model_path}/w2v.m')

    #提取出每个文章的向量
    article_vec_map = {}
    for word in set(words):
        if word in model.wv:
            article_vec_map[int(word)] = model.wv[word]

    return article_vec_map


@multitasking.task
def recall(df_query, article_vec_map, article_index, user_item_dict,
           worker_id):
    """
    df_query: 要给出的用户的喜欢物品
    article_vec_map: 每个文章的向量表
    article_index: 构建的Annoy索引
    user_item_dict: 用户的交互历史
    """
    data_list = []

    for user_id, item_id in tqdm(df_query.values):
        #没有交互过，即没有交互过任何物品，则跳过此用户的推荐过程。
        if user_id not in user_item_dict:
            continue
        
        rank = defaultdict(int)

        # 最后一次点击召回， 实时性：反映用户最新兴趣
        interacted_items = user_item_dict[user_id]
        interacted_items = interacted_items[-1:]

        for item in interacted_items:
            article_vec = article_vec_map[item]

            #返回100个最近的向量，distance：距离小 → 相似度大
            item_ids, distances = article_index.get_nns_by_vector(
                article_vec, 100, include_distances=True)
            
            #用于计算转化相似度
            sim_scores = [2 - distance for distance in distances]

            for relate_item, wij in zip(item_ids, sim_scores):
                if relate_item not in interacted_items:
                    rank.setdefault(relate_item, 0)
                    rank[relate_item] += wij
        

        #排序后选择前50个相似的物品
        sim_items = sorted(rank.items(), key=lambda d: d[1], reverse=True)[:50]

        #分别取出相似的物品id，以及相似度
        item_ids = [item[0] for item in sim_items]
        item_sim_scores = [item[1] for item in sim_items]

        df_temp = pd.DataFrame()
        df_temp['article_id'] = item_ids
        df_temp['sim_score'] = item_sim_scores
        df_temp['user_id'] = user_id

        if item_id == -1:
            #说明是测试集中的数据
            df_temp['label'] = np.nan
        else:
            #说明是验证集中的数据
            df_temp['label'] = 0
            #等于要验证的结果的情况
            df_temp.loc[df_temp['article_id'] == item_id, 'label'] = 1

        df_temp = df_temp[['user_id', 'article_id', 'sim_score', 'label']]
        df_temp['user_id'] = df_temp['user_id'].astype('int')
        df_temp['article_id'] = df_temp['article_id'].astype('int')

        data_list.append(df_temp)

    df_data = pd.concat(data_list, sort=False)

    os.makedirs('/home/wangtiantian/shikainan/newscommemder/top2wk/user_data/tmp/w2v', exist_ok=True)
    df_data.to_pickle('/home/wangtiantian/shikainan/newscommemder/top2wk/user_data/tmp/w2v/{}.pkl'.format(worker_id))


if __name__ == '__main__':

    if mode == 'valid':
        df_click = pd.read_pickle('/home/wangtiantian/shikainan/newscommemder/top2wk/user_data/data/offline/click.pkl')
        df_query = pd.read_pickle('/home/wangtiantian/shikainan/newscommemder/top2wk/user_data/data/offline/query.pkl')

        os.makedirs('/home/wangtiantian/shikainan/newscommemder/top2wk/user_data/data/offline', exist_ok=True)
        os.makedirs('/home/wangtiantian/shikainan/newscommemder/top2wk/user_data/model/offline', exist_ok=True)

        w2v_file = '/home/wangtiantian/shikainan/newscommemder/top2wk/user_data/data/offline/article_w2v.pkl'
        model_path = '/home/wangtiantian/shikainan/newscommemder/top2wk/user_data/model/offline'
    else:
        df_click = pd.read_pickle('/home/wangtiantian/shikainan/newscommemder/top2wk/user_data/data/online/click.pkl')
        df_query = pd.read_pickle('/home/wangtiantian/shikainan/newscommemder/top2wk/user_data/data/online/query.pkl')

        os.makedirs('/home/wangtiantian/shikainan/newscommemder/top2wk/user_data/data/online', exist_ok=True)
        os.makedirs('/home/wangtiantian/shikainan/newscommemder/top2wk/user_data/model/online', exist_ok=True)

        w2v_file = '/home/wangtiantian/shikainan/newscommemder/top2wk/user_data/data/online/article_w2v.pkl'
        model_path = '/home/wangtiantian/shikainan/newscommemder/top2wk/user_data/model/online'

    log.debug(f'df_click shape: {df_click.shape}')
    log.debug(f'{df_click.head()}')

    #计算每个文章的向量，并进行保存
    article_vec_map = word2vec(df_click, 'user_id', 'click_article_id',
                               model_path)
    f = open(w2v_file, 'wb')
    pickle.dump(article_vec_map, f)
    f.close()
    

    # 将 embedding 建立索引
    article_index = AnnoyIndex(256, 'angular')
    article_index.set_seed(2020)

    # 第一部分：构建Annoy索引
    for article_id, emb in tqdm(article_vec_map.items()):
        article_index.add_item(article_id, emb)

    # 100: 构建的随机投影树的数量
    # 更多的树 → 更高的查找精度，但构建时间更长，内存占用更大
    # 更少的树 → 更快的构建速度，但查找精度稍低
    article_index.build(100)

    # 第二部分：构建用户点击历史字典
    user_item_ = df_click.groupby('user_id')['click_article_id'].agg(
        lambda x: list(x)).reset_index()
    user_item_dict = dict(
        zip(user_item_['user_id'], user_item_['click_article_id']))

    # 召回，首先划分每一个线程要查找多少
    n_split = max_threads
    all_users = df_query['user_id'].unique()
    shuffle(all_users)
    total = len(all_users)
    n_len = total // n_split

    # 清空临时文件夹
    for path, _, file_list in os.walk('/home/wangtiantian/shikainan/newscommemder/top2wk/user_data/tmp/w2v'):
        for file_name in file_list:
            os.remove(os.path.join(path, file_name))

    for i in range(0, total, n_len):
        part_users = all_users[i:i + n_len]
        df_temp = df_query[df_query['user_id'].isin(part_users)]
        recall(df_temp, article_vec_map, article_index, user_item_dict, i)

    multitasking.wait_for_tasks()
    log.info('合并任务')

    df_data = pd.DataFrame()
    for path, _, file_list in os.walk('/home/wangtiantian/shikainan/newscommemder/top2wk/user_data/tmp/w2v'):
        for file_name in file_list:
            # df_temp = pd.read_pickle(os.path.join(path, file_name))
            # df_data = df_data.append(df_temp)

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

        #筛选出验证集
        total = df_query[df_query['click_article_id'] != -1].user_id.nunique()

        hitrate_5, mrr_5, hitrate_10, mrr_10, hitrate_20, mrr_20, hitrate_40, mrr_40, hitrate_50, mrr_50 = evaluate(
            df_data[df_data['label'].notnull()], total)

        log.debug(
            f'w2v: {hitrate_5}, {mrr_5}, {hitrate_10}, {mrr_10}, {hitrate_20}, {mrr_20}, {hitrate_40}, {mrr_40}, {hitrate_50}, {mrr_50}'
        )
    # 保存召回结果
    if mode == 'valid':
        df_data.to_pickle('/home/wangtiantian/shikainan/newscommemder/top2wk/user_data/data/offline/recall_w2v.pkl')
    else:
        df_data.to_pickle('/home/wangtiantian/shikainan/newscommemder/top2wk/user_data/data/online/recall_w2v.pkl')
