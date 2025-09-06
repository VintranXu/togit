import argparse
import os
import random
from random import sample

import pandas as pd
from tqdm import tqdm

from utils import Logger

random.seed(2020)

# 命令行参数
parser = argparse.ArgumentParser(description='数据处理')
parser.add_argument('--mode', default='valid')
parser.add_argument('--logfile', default='test.log')

args = parser.parse_args()

mode = args.mode
logfile = args.logfile

# 初始化日志
os.makedirs('/home/wangtiantian/shikainan/newscommemder/top2wk/user_data/log', exist_ok=True)
log = Logger(f'/home/wangtiantian/shikainan/newscommemder/top2wk/user_data/log/{logfile}').logger
log.info(f'数据处理，mode: {mode}')


def data_offline(df_train_click, df_test_click):
    # 从训练集的用户列表中随机抽取 50,000 个用户作为验证集。
    # 取出训练集用户id
    train_users = df_train_click['user_id'].values.tolist()
    # 随机采样出一部分样本
    val_users = sample(train_users, 50000)
    log.debug(f'val_users num: {len(set(val_users))}')

    
    #构建训练集和验证集
    click_list = []
    valid_query_list = []

    groups = df_train_click.groupby(['user_id'])

    for user_id, g in tqdm(groups):
        if user_id[0] in val_users:

            #如果某个验证集用户只有1条点击记录：

            # g.tail(1) 取到这唯一的记录放入 df_query
            # g.head(g.shape[0] - 1) = g.head(0) 返回空的DataFrame
            # 这个用户的0条历史记录被加入训练数据
            valid_query = g.tail(1)
            valid_query_list.append(
                valid_query[['user_id', 'click_article_id']])


            #训练集获取除最后一条以外的数据
            train_click = g.head(g.shape[0] - 1)
            click_list.append(train_click)
        else:
            #全部用来训练
            click_list.append(g)
    
    print(len(click_list), len(valid_query_list))

    #重新转化为pd格式
    df_train_click = pd.concat(click_list, sort=False)
    df_valid_query = pd.concat(valid_query_list, sort=False)

    #获取测试集用户
    test_users = df_test_click['user_id'].unique()
    #构建测试集用户的推荐结果
    test_query_list = []
    
    
    for user in tqdm(test_users):
        test_query_list.append([user, -1])

    #转化为pd格式
    df_test_query = pd.DataFrame(test_query_list,
                                 columns=['user_id', 'click_article_id'])

    df_query = pd.concat([df_valid_query, df_test_query],
                         sort=False).reset_index(drop=True)
    df_click = pd.concat([df_train_click, df_test_click],
                         sort=False).reset_index(drop=True)
    df_click = df_click.sort_values(['user_id',
                                     'click_timestamp']).reset_index(drop=True)

    log.debug(
        f'df_query shape: {df_query.shape}, df_click shape: {df_click.shape}')
    log.debug(f'{df_query.head()}')
    log.debug(f'{df_click.head()}')

    # 保存文件
    os.makedirs('/home/wangtiantian/shikainan/newscommemder/top2wk/user_data/data/offline', exist_ok=True)

    df_click.to_pickle('/home/wangtiantian/shikainan/newscommemder/top2wk/user_data/data/offline/click.pkl')
    df_query.to_pickle('/home/wangtiantian/shikainan/newscommemder/top2wk/user_data/data/offline/query.pkl')


def data_online(df_train_click, df_test_click):
    test_users = df_test_click['user_id'].unique()
    test_query_list = []

    for user in tqdm(test_users):
        test_query_list.append([user, -1])

    df_test_query = pd.DataFrame(test_query_list,
                                 columns=['user_id', 'click_article_id'])


    df_query = df_test_query
    #在线数据，推全
    df_click = pd.concat([df_train_click, df_test_click],
                         sort=False).reset_index(drop=True)
    
    df_click = df_click.sort_values(['user_id',
                                     'click_timestamp']).reset_index(drop=True)

    log.debug(
        f'df_query shape: {df_query.shape}, df_click shape: {df_click.shape}')
    log.debug(f'{df_query.head()}')
    log.debug(f'{df_click.head()}')

    # 保存文件
    os.makedirs('/home/wangtiantian/shikainan/newscommemder/top2wk/data/online', exist_ok=True)

    df_click.to_pickle('/home/wangtiantian/shikainan/newscommemder/top2wk/user_data/data/online/click.pkl')
    df_query.to_pickle('/home/wangtiantian/shikainan/newscommemder/top2wk/user_data/data/online/query.pkl')


if __name__ == '__main__':
    df_train_click = pd.read_csv('/home/wangtiantian/shikainan/newswork/data/train_click_log.csv')
    df_test_click = pd.read_csv('/home/wangtiantian/shikainan/newswork/data/testA_click_log.csv')
    os.makedirs('/home/wangtiantian/shikainan/newscommemder/top2wk/user_data/data/offline', exist_ok=True)

    log.debug(
        f'df_train_click shape: {df_train_click.shape}, df_test_click shape: {df_test_click.shape}'
    )

    if mode == 'valid':
        data_offline(df_train_click, df_test_click)
    else:
        data_online(df_train_click, df_test_click)
