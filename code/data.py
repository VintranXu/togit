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
os.makedirs('./user_data/log', exist_ok=True)
log = Logger(f'./user_data/log/{logfile}').logger
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

    # ========== 重新转化为 DataFrame 格式 ==========
    
    # ========== df_train_click 包含什么？==========
    # 训练集的点击历史数据，用于训练模型
    # 内容:
    #   1. 验证集用户的历史记录 (除去最后1条)
    #   2. 非验证集用户的所有记录
    # 列: ['user_id', 'click_article_id', 'click_timestamp', ...]
    # 
    # 示例:
    #   user_id  click_article_id  click_timestamp
    #   1001     5001             100    <- 验证用户的历史
    #   1001     5003             102    <- 验证用户的历史
    #   1002     5002             101    <- 训练用户的全部
    #   1002     5005             104    <- 训练用户的全部
    # 
    # 用途: 作为模型训练时的历史行为数据
    df_train_click = pd.concat(click_list, sort=False)
    
    # ========== df_valid_query 包含什么？==========
    # 验证集的查询目标，用于评估模型召回效果
    # 内容: 验证集用户的最后1条点击记录 (ground truth)
    # 列: ['user_id', 'click_article_id']
    #
    # 示例:
    #   user_id  click_article_id
    #   1001     5004            <- 验证用户1001的真实点击
    #   1003     5008            <- 验证用户1003的真实点击
    #   1005     5012            <- 验证用户1005的真实点击
    #
    # 用途: 评估召回结果，看模型能否召回这些真实点击的物品
    df_valid_query = pd.concat(valid_query_list, sort=False)

    # ========== 构建测试集查询 ==========
    # 获取测试集用户（testA数据中的用户）
    test_users = df_test_click['user_id'].unique()
    # 构建测试集用户的推荐查询
    # 注意: click_article_id = -1 表示测试集没有真实标签（需要预测）
    test_query_list = []
    
    for user in tqdm(test_users):
        test_query_list.append([user, -1])

    # ========== df_test_query 包含什么？==========
    # 测试集的查询请求，用于生成最终提交结果
    # 内容: 测试集用户列表，click_article_id=-1 表示未知（需要预测）
    # 列: ['user_id', 'click_article_id']
    #
    # 示例:
    #   user_id  click_article_id
    #   2001     -1              <- 需要为用户2001预测推荐列表
    #   2002     -1              <- 需要为用户2002预测推荐列表
    #
    # 用途: 生成最终提交的推荐结果
    df_test_query = pd.DataFrame(test_query_list,
                                 columns=['user_id', 'click_article_id'])

    # ========== df_query 包含什么？为什么要加测试集？==========
    # 作用: 定义"需要为哪些用户生成推荐"的查询列表
    # 
    # 内容:
    #   - 验证集查询 (df_valid_query): click_article_id = 真实点击的物品ID
    #   - 测试集查询 (df_test_query): click_article_id = -1 (未知)
    #
    # 为什么要加测试集？
    #   1. 最终目的是为测试集用户生成推荐结果并提交
    #   2. 验证集用于离线评估模型效果（有真实标签可以计算指标）
    #   3. 一次性处理所有查询，提高效率
    #
    # 召回流程:
    #   输入: df_query (用户列表)
    #   处理: 对每个用户，基于历史生成Top-N推荐
    #   输出: 推荐列表
    #   评估: 验证集用户 -> 计算Hit@K, MRR等指标
    #        测试集用户 -> 生成提交文件
    #
    # 示例:
    #   user_id  click_article_id
    #   1001     5006            <- 验证: 看推荐中是否包含5006
    #   1002     5010            <- 验证: 看推荐中是否包含5010
    #   2001     -1              <- 测试: 生成推荐列表提交
    #   2002     -1              <- 测试: 生成推荐列表提交
    df_query = pd.concat([df_valid_query, df_test_query],
                         sort=False).reset_index(drop=True)
    
    # ========== df_click 包含什么？为什么要加测试集？==========
    # 作用: 提供"所有用户的完整历史行为"用于构建用户画像
    #
    # 内容:
    #   - 训练集处理后的历史 (df_train_click)
    #   - 测试集用户的历史 (df_test_click)
    #
    # 为什么要加测试集？
    #   1. 测试集用户也有历史点击记录！
    #   2. 召回时需要根据用户历史生成推荐
    #      例: 用户2001历史看过[1001, 1005, 1008]
    #          -> 基于这些历史推荐相似物品
    #   3. 如果不加测试集历史，测试集用户就没有历史数据，无法生成推荐
    #
    # 召回时如何使用:
    #   - YouTube DNN: histories = df_click中该用户的点击记录
    #   - ItemCF: 基于该用户点击过的物品推荐相似物品
    #   - Word2Vec: 基于用户点击序列训练embedding
    #
    # 示例数据:
    #   user_id  click_article_id  click_timestamp
    #   1001     5001             100    <- 验证用户的历史
    #   1001     5003             102    
    #   2001     1001             200    <- 测试用户的历史
    #   2001     1005             201    <- 测试用户的历史
    #   2001     1008             202    <- 测试用户的历史
    #
    # 关键理解: 
    #   - df_train_click: 用于"训练模型"的历史
    #   - df_click: 用于"召回推理"时的完整历史（训练+测试）
    df_click = pd.concat([df_train_click, df_test_click],
                         sort=False).reset_index(drop=True)
    df_click = df_click.sort_values(['user_id',
                                     'click_timestamp']).reset_index(drop=True)

    log.debug(
        f'df_query shape: {df_query.shape}, df_click shape: {df_click.shape}')
    log.debug(f'{df_query.head()}')
    log.debug(f'{df_click.head()}')

    # 保存文件
    os.makedirs('./user_data/data/offline', exist_ok=True)

    df_click.to_pickle('./user_data/data/offline/click.pkl')
    df_query.to_pickle('./user_data/data/offline/query.pkl')


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
    os.makedirs('./user_data/data/online', exist_ok=True)

    df_click.to_pickle('./user_data/data/online/click.pkl')
    df_query.to_pickle('./user_data/data/online/query.pkl')


if __name__ == '__main__':
    df_train_click = pd.read_csv('./data/train_click_log.csv')
    df_test_click = pd.read_csv('./data/testA_click_log.csv')
    os.makedirs('./user_data/data/offline', exist_ok=True)

    log.debug(
        f'df_train_click shape: {df_train_click.shape}, df_test_click shape: {df_test_click.shape}'
    )

    if mode == 'valid':
        data_offline(df_train_click, df_test_click)
    else:
        data_online(df_train_click, df_test_click)
