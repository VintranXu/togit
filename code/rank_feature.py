import argparse
import os
import pickle
import warnings

import numpy as np
import pandas as pd
from pandarallel import pandarallel

from utils import Logger

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

pandarallel.initialize()

warnings.filterwarnings('ignore')

seed = 2020

# 命令行参数
parser = argparse.ArgumentParser(description='排序特征')
parser.add_argument('--mode', default='valid')
parser.add_argument('--logfile', default='test.log')

args = parser.parse_args()

mode = args.mode
logfile = args.logfile

# 初始化日志
os.makedirs('/home/wangtiantian/shikainan/newscommemder/top2wk/user_data/log', exist_ok=True)
log = Logger(f'/home/wangtiantian/shikainan/newscommemder/top2wk/user_data/log/{logfile}').logger
log.info(f'排序特征，mode: {mode}')



def func_if_sum(x):
    user_id = x['user_id']
    article_id = x['article_id']

    #取出用户的交互信息
    interacted_items = user_item_dict[user_id]

    #将用户的点击历史反转
    interacted_items = interacted_items[::-1]

    # 计算当前候选文章 article_id 与该用户所有历史点击文章的ItemCF相似度之和，并且这个“和”是加权的。
    sim_sum = 0
    for loc, i in enumerate(interacted_items):
        try:
            sim_sum += item_sim[i][article_id] * (0.7**loc)
        except Exception as e:
            pass
    return sim_sum


def func_if_last(x):
    user_id = x['user_id']
    article_id = x['article_id']

    last_item = user_item_dict[user_id][-1]

    sim = 0
    try:
        #计算出当前候选文章与作者最后一次点击文章的相似度
        sim = item_sim[last_item][article_id]
    except Exception as e:
        pass
    return sim


# def func_binetwork_sim_last(x):
#     user_id = x['user_id']
#     article_id = x['article_id']

#     last_item = user_item_dict[user_id][-1]

#     sim = 0
#     try:
#         sim = binetwork_sim[last_item][article_id]
#     except Exception as e:
#         pass
#     return sim


def consine_distance(vector1, vector2):
    if type(vector1) != np.ndarray or type(vector2) != np.ndarray:
        return -1
    distance = np.dot(vector1, vector2) / \
        (np.linalg.norm(vector1)*(np.linalg.norm(vector2)))
    return distance

#计算相似度
def func_w2w_sum(x, num):
    user_id = x['user_id']
    article_id = x['article_id']

    interacted_items = user_item_dict[user_id]
    interacted_items = interacted_items[::-1][:num]

    sim_sum = 0
    for loc, i in enumerate(interacted_items):
        try:
            sim_sum += consine_distance(article_vec_map[article_id],
                                        article_vec_map[i])
        except Exception as e:
            pass
    return sim_sum


def func_w2w_last_sim(x):
    user_id = x['user_id']
    article_id = x['article_id']

    last_item = user_item_dict[user_id][-1]

    sim = 0
    try:
        sim = consine_distance(article_vec_map[article_id],
                               article_vec_map[last_item])
    except Exception as e:
        pass
    return sim


if __name__ == '__main__':
    if mode == 'valid':
        df_feature = pd.read_pickle('/home/wangtiantian/shikainan/newscommemder/top2wk/user_data/data/offline/recall.pkl')
        df_click = pd.read_pickle('/home/wangtiantian/shikainan/newscommemder/top2wk/user_data/data/offline/click.pkl')

    else:
        df_feature = pd.read_pickle('/home/wangtiantian/shikainan/newscommemder/top2wk/user_data/data/online/recall.pkl')
        df_click = pd.read_pickle('/home/wangtiantian/shikainan/newscommemder/top2wk/user_data/data/online/click.pkl')


    # 文章特征
    log.debug(f'df_feature.shape: {df_feature.shape}')

    # 打印前5行数据
    print(df_feature.head())
    
    # df_click: [user_id, click_article_id, timestamp]
    # df_feature: [user_id, article_id, label, sim_score]
    # df_article: [article_id,  category_id, created_at_ts, words_count]
    df_article = pd.read_csv('/home/wangtiantian/shikainan/newswork/data/articles.csv')


    #将时间戳除以1000
    df_article['created_at_ts'] = df_article['created_at_ts'] / 1000
    df_article['created_at_ts'] = df_article['created_at_ts'].astype('int')

    #将文章的信息合并起来 df_feature: [user_id, article_id, label, sim_score, category_id, created_at_ts, words_count]
    df_feature = df_feature.merge(df_article, how='left')

    #再将ms表示的时间戳转化为s表示的时间戳后变成真正的时间:  df_feature: [user_id, article_id, label, sim_score, category_id, created_at_ts, words_count, created_at_datetime]
    df_feature['created_at_datetime'] = pd.to_datetime(
        df_feature['created_at_ts'], unit='s')

    #打印操作
    log.debug(f'df_article.head(): {df_article.head()}')
    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')


    # 历史记录相关特征
    # 对用户的点击日志进行排序。 df_click: [user_id, click_article_id, timestamp]
    df_click.sort_values(['user_id', 'click_timestamp'], inplace=True)

    #改名 
    # df_click: [user_id, article_id, timestamp]
    # df_article: [article_id,  category_id, created_at_ts, words_count]

    df_click.rename(columns={'click_article_id': 'article_id'}, inplace=True)

    #合并后为 df_click: [user_id, article_id, timestamp, category_id, created_at_ts, words_count]
    df_click = df_click.merge(df_article, how='left')
    # 将点击时间戳的单位从毫秒转换为秒。
    df_click['click_timestamp'] = df_click['click_timestamp'] / 1000
    # 将整数形式的点击时间戳，转换为人类可读的标准日期时间格式。
    df_click['click_datetime'] = pd.to_datetime(df_click['click_timestamp'],
                                                unit='s',
                                                errors='coerce')
    # 从日期时间对象中提取小时信息。
    # df_click: [user_id, article_id, timestamp, category_id, created_at_ts, words_count, click_datetime_hour]
    df_click['click_datetime_hour'] = df_click['click_datetime'].dt.hour

    # 用户点击文章的创建时间差的平均值

    # 计算每一次点击与其上一次点击的文章创作时间差
    df_click['user_id_click_article_created_at_ts_diff'] = df_click.groupby(
        ['user_id'])['created_at_ts'].diff()
    
    # 得到均值
    df_temp = df_click.groupby([
        'user_id'
    ])['user_id_click_article_created_at_ts_diff'].mean().reset_index()

    # 为这个临时 DataFrame df_temp 的列进行重命名，使其更具可读性。
    df_temp.columns = [
        'user_id', 'user_id_click_article_created_at_ts_diff_mean'
    ]

    # 合并, df_feature: [user_id, article_id, label, sim_score, category_id, created_at_ts, words_count, created_at_datetime, user_id_click_article_created_at_ts_diff_mean]
    """
    值较大 (远大于0): 说明这个用户倾向于越看越新的文章。他可能是一个追热点、关心时事的用户。
    值接近0: 说明用户阅读的文章创作时间比较随机，没有明显的时效性偏好。
    值为负: 说明这个用户倾向于回顾旧闻或者阅读深度、非时效性的内容。他可能是一个喜欢看专题、历史或科普类内容的用户。
    """
    df_feature = df_feature.merge(df_temp, how='left')

    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')
    

    # 用户点击文章的时间差的平均值
    df_click['user_id_click_diff'] = df_click.groupby(
        ['user_id'])['click_timestamp'].diff()
    df_temp = df_click.groupby(['user_id'
                                ])['user_id_click_diff'].mean().reset_index()
    df_temp.columns = ['user_id', 'user_id_click_diff_mean']

    #  df_feature: [user_id, article_id, label, sim_score, category_id, created_at_ts, words_count, created_at_datetime, user_id_click_article_created_at_ts_diff_mean, user_id_click_diff_mean]
    df_feature = df_feature.merge(df_temp, how='left')

    """
    值较小: 说明这个用户阅读速度快，点击频繁。他可能是一个正在“刷新闻”的活跃用户，或者正在快速浏览标题以寻找感兴趣的内容。
    值较大: 说明这个用户阅读得更仔细，或者是非连续性阅读。他可能花更多时间在每一篇文章上，或者是在不同时间段零散地阅读新闻。
    """
    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')
    

    # 点击文章与创建时间差的统计值
    # df_feature: [user_id, article_id, label, sim_score, category_id, created_at_ts, words_count, created_at_datetime,  + 
    #  + user_id_click_article_created_at_ts_diff_mean(用户点击文章创建时间平均值), user_id_click_diff_mean(续点击的时间间隔), user_click_timestamp_created_at_ts_diff_mean(平均新鲜度), user_click_timestamp_created_at_ts_diff_std(新鲜度稳定性)]
    """
    user_click_timestamp_created_at_ts_diff_mean (平均新鲜度)
    这个特征直接描绘了用户的画像：
    值很小：说明这个用户阅读的文章，平均来看都是刚发布不久的。这很可能是一个**“新闻控”或“热点追逐者”** 📰。他/她非常关心实时动态。
    值很大：说明这个用户阅读的文章，平均来看都发布了很久。他/她可能不关心时效性，更喜欢阅读深度报道、专题、科普或“常青”内容 📚。

    2. user_click_timestamp_created_at_ts_diff_std (新鲜度稳定性)
    这个特征从另一个维度补充了用户信息：
    值很小：说明这个用户的阅读偏好非常稳定和专一。如果他的平均值很小，那么他几乎只看新文章；如果平均值很大，那他几乎只看旧文章。
    值很大：说明这个用户的阅读偏好非常广泛和多变。他可能一会儿看看热点新闻，一会儿又去翻翻几天前的专题报道，时效性偏好不固定。
    """
    df_click['click_timestamp_created_at_ts_diff'] = df_click[
        'click_timestamp'] - df_click['created_at_ts']

    
    # df_temp = df_click.groupby(
    #     ['user_id'])['click_timestamp_created_at_ts_diff'].agg({
    #         'user_click_timestamp_created_at_ts_diff_mean':
    #         'mean',
    #         'user_click_timestamp_created_at_ts_diff_std':
    #         'std'
    #     }).reset_index()

    df_temp = df_click.groupby('user_id').agg(
        user_click_timestamp_created_at_ts_diff_mean=('click_timestamp_created_at_ts_diff', 'mean'),
        user_click_timestamp_created_at_ts_diff_std=('click_timestamp_created_at_ts_diff', 'std')
    ).reset_index()

    df_feature = df_feature.merge(df_temp, how='left')

    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')

    

    # 点击的新闻的 click_datetime_hour 统计值
    """
    构建一个描述用户阅读时间习惯的特征。具体来说，它通过计算用户所有点击行为发生在一天中不同小时的标准差（Standard Deviation），来衡量用户的阅读时间是**“集中”还是“分散”**。
    """
    # df_temp = df_click.groupby(['user_id'])['click_datetime_hour'].agg({
    #     'user_click_datetime_hour_std':
    #     'std'
    # }).reset_index()

    df_temp = df_click.groupby('user_id').agg(
        user_click_datetime_hour_std=('click_datetime_hour', 'std')
    ).reset_index()
    """
     df_feature: [user_id, article_id, label, sim_score, category_id, created_at_ts, words_count, created_at_datetime,  
     user_id_click_article_created_at_ts_diff_mean(用户点击文章创建时间平均值), user_id_click_diff_mean(续点击的时间间隔),
     user_click_timestamp_created_at_ts_diff_mean(平均新鲜度), user_click_timestamp_created_at_ts_diff_std(新鲜度稳定性)
     user_click_datetime_hour_std(点击的标准差)
    """

    df_feature = df_feature.merge(df_temp, how='left')

    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')



    # 点击的新闻的 words_count 统计值,  user_clicked_article_words_count_mean:所有点击的文章的平均下来词的数量, 上一次点击的文本长度
    """
    user_clicked_article_words_count_mean (长期平均偏好)
    这个特征刻画了用户稳定、长期的阅读习惯：
    值很大: 说明该用户长期来看偏爱长篇文章、深度报道、专题分析等。他可能是一个喜欢沉浸式阅读、追求信息深度的用户。
    值很小: 说明该用户长期来看偏爱短篇资讯、快讯、简报等。他可能是一个喜欢快速获取信息、浏览标题的用户。

    2. user_click_last_article_words_count (近期瞬时偏好)
    这个特征则捕捉了用户当前的阅读兴趣，非常重要：
    上下文感知: 用户的阅读兴趣是会变化的。他可能平时喜欢看短新闻，但上一次点击恰好是一篇长篇报道。这个特征就能捕捉到这种即时的兴趣漂移。
    会话连续性: 用户的下一次点击行为，有很大概率会和他上一次的行为相关。如果用户上一篇看的是长文，那么下一篇推荐长文的成功率可能会更高。
    """
    # df_temp = df_click.groupby(['user_id'])['words_count'].agg({
    #     'user_clicked_article_words_count_mean':
    #     'mean',
    #     'user_click_last_article_words_count':
    #     lambda x: x.iloc[-1]
    # }).reset_index()

    df_temp = df_click.groupby('user_id').agg(
        user_clicked_article_words_count_mean=('words_count', 'mean'),
        user_click_last_article_words_count=('words_count', lambda x: x.iloc[-1])
    ).reset_index()

    """
     df_feature: [user_id, article_id, label, sim_score, category_id, created_at_ts, words_count, created_at_datetime,  
     user_id_click_article_created_at_ts_diff_mean(用户点击文章创建时间平均值), user_id_click_diff_mean(续点击的时间间隔),
     user_click_timestamp_created_at_ts_diff_mean(平均新鲜度), user_click_timestamp_created_at_ts_diff_std(新鲜度稳定性)
     user_click_datetime_hour_std(点击的标准差),  user_clicked_article_words_count_mean(长期阅读文本的篇幅), user_click_last_article_words_count(上次阅读文本的篇幅)
    """
    df_feature = df_feature.merge(df_temp, how='left')

    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')





    # 点击的新闻的 created_at_ts 统计值
    """
    user_click_last_article_created_time (最近行为锚点)
    这个特征记录了用户最近一次点击的文章是什么时候创作的。它代表了用户当前兴趣的上下文。在后续的代码中，会立刻用它来构建交叉特征，例如：
    候选文章的创作时间 - user_click_last_article_created_time

    这个差值特征非常有威力，它直接衡量了“这篇推荐的文章是比我上一篇看的要新，还是旧？”。这对于预测用户是否会因为文章的新颖性或相关性而点击至关重要。

    user_clicked_article_created_time_max (历史偏好极值)
    这个特征记录了用户历史上看过的“最新”的文章是什么时候创作的。它代表了用户历史兴趣范围的上界**。

    区分 last 和 max: 在上面的例子中，last 是 120，而 max 是 150。这说明用户最近一次点击的文章（创作于120）并不是他历史上看过的最新的一篇（他曾经看过一篇创作于150的文章）。这种情况很常见，比如用户回顾了一篇旧闻。
    价值: 这个特征可以用来构建“候选文章是否突破了用户的历史新颖度记录”这样的特征。如果一篇候选文章比用户历史上看过的所有文章都新，那么它可能因为“前所未见”的新颖性而具有额外的吸引力。
    """
    # df_temp = df_click.groupby('user_id')['created_at_ts'].agg({
    #     'user_click_last_article_created_time':
    #     lambda x: x.iloc[-1],
    #     'user_clicked_article_created_time_max':
    #     'max',
    # }).reset_index()

    df_temp = df_click.groupby('user_id').agg(
        user_click_last_article_created_time=('created_at_ts', lambda x: x.iloc[-1]),
        user_clicked_article_created_time_max=('created_at_ts', 'max')
    ).reset_index()

    """
     df_feature: [user_id, article_id, label, sim_score, category_id, created_at_ts, words_count, created_at_datetime,  
     user_id_click_article_created_at_ts_diff_mean(用户点击文章创建时间平均值), user_id_click_diff_mean(续点击的时间间隔),
     user_click_timestamp_created_at_ts_diff_mean(平均新鲜度), user_click_timestamp_created_at_ts_diff_std(新鲜度稳定性)
     user_click_datetime_hour_std(点击的标准差),  user_clicked_article_words_count_mean(长期阅读文本的篇幅), user_click_last_article_words_count(上次阅读文本的篇幅)
     user_click_last_article_created_time(上次点击文章的的创建时间), user_clicked_article_created_time_max(浏览过最新的文章的创建时间是什么)
    """
    df_feature = df_feature.merge(df_temp, how='left')

    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')





    # 点击的新闻的 click_timestamp 统计值
    """
    'user_click_last_article_click_time': lambda x: x.iloc[-1]: 获取每个用户最近一次点击行为发生的时间戳。这代表了用户最后一次在App内活跃的时间点，是重要的上下文信息。
    'user_clicked_article_click_time_mean': 'mean': 计算每个用户所有历史点击时间的平均时间戳。这个特征的直接意义不大，但可以作为一个基准值，反映用户历史活跃时间的中心点。
    """

    # df_temp = df_click.groupby('user_id')['click_timestamp'].agg({
    #     'user_click_last_article_click_time':
    #     lambda x: x.iloc[-1],
    #     'user_clicked_article_click_time_mean':
    #     'mean',
    # }).reset_index()

    df_temp = df_click.groupby('user_id').agg(
        user_click_last_article_click_time=('click_timestamp', lambda x: x.iloc[-1]),
        user_clicked_article_click_time_mean=('click_timestamp', 'mean')
    ).reset_index()

    """
     df_feature: [user_id, article_id, label, sim_score, category_id, created_at_ts, words_count, created_at_datetime,  
     user_id_click_article_created_at_ts_diff_mean(用户点击文章创建时间平均值), user_id_click_diff_mean(续点击的时间间隔),
     user_click_timestamp_created_at_ts_diff_mean(平均新鲜度), user_click_timestamp_created_at_ts_diff_std(新鲜度稳定性)
     user_click_datetime_hour_std(点击的标准差),  user_clicked_article_words_count_mean(长期阅读文本的篇幅), user_click_last_article_words_count(上次阅读文本的篇幅)
     user_click_last_article_created_time(上次点击文章的的创建时间), user_clicked_article_created_time_max(浏览过最新的文章的创建时间是什么)
     user_click_last_article_click_time(上次点击文章的时间), user_clicked_article_click_time_mean(点击文章的平均时间戳)
    """
    df_feature = df_feature.merge(df_temp, how='left')

    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')


    # 交叉特征
    # 计算候选文章与用户上一次行为的“时间差”特征
    df_feature['user_last_click_created_at_ts_diff'] = df_feature[
        'created_at_ts'] - df_feature['user_click_last_article_created_time']
    
    # 计算候选文章与用户上一次行为的“新鲜度差”特征
    df_feature['user_last_click_timestamp_diff'] = df_feature[
        'created_at_ts'] - df_feature['user_click_last_article_click_time']
    
    # 计算候选文章与用户上一次行为的“篇幅差”特征
    df_feature['user_last_click_words_count_diff'] = df_feature[
        'words_count'] - df_feature['user_click_last_article_words_count']
    
    """
     df_feature: [user_id, article_id, label, sim_score, category_id, created_at_ts, words_count, created_at_datetime,  
     user_id_click_article_created_at_ts_diff_mean(用户点击文章创建时间平均值), user_id_click_diff_mean(续点击的时间间隔),
     user_click_timestamp_created_at_ts_diff_mean(平均新鲜度), user_click_timestamp_created_at_ts_diff_std(新鲜度稳定性)
     user_click_datetime_hour_std(点击的标准差),  user_clicked_article_words_count_mean(长期阅读文本的篇幅), user_click_last_article_words_count(上次阅读文本的篇幅)
     user_click_last_article_created_time(上次点击文章的的创建时间), user_clicked_article_created_time_max(浏览过最新的文章的创建时间是什么)
     user_click_last_article_click_time(上次点击文章的时间), user_clicked_article_click_time_mean(点击文章的平均时间戳)
     user_last_click_created_at_ts_diff(当前文章的创建时间与上次点击文章创建时间之差), user_last_click_timestamp_diff(当前文章的创建时间与上次点击文章的时间之差), user_last_click_words_count_diff(当前文章的字数与上次点击文章的字数之差)

    """

    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')


    # 计数统计, 用户的活跃信息(user_id_cnt),  文章的热门信息(article_id_cnt), 用户对某个类别的兴趣(user_id_category_id_cnt)
    for f in [['user_id'], ['article_id'], ['user_id', 'category_id']]:
        df_temp = df_click.groupby(f).size().reset_index()
        df_temp.columns = f + ['{}_cnt'.format('_'.join(f))]

        df_feature = df_feature.merge(df_temp, how='left')

    """
     df_feature: [user_id, article_id, label, sim_score, category_id, created_at_ts, words_count, created_at_datetime,  
     user_id_click_article_created_at_ts_diff_mean(用户点击文章创建时间平均值), user_id_click_diff_mean(续点击的时间间隔),
     user_click_timestamp_created_at_ts_diff_mean(平均新鲜度), user_click_timestamp_created_at_ts_diff_std(新鲜度稳定性)
     user_click_datetime_hour_std(点击的标准差),  user_clicked_article_words_count_mean(长期阅读文本的篇幅), user_click_last_article_words_count(上次阅读文本的篇幅)
     user_click_last_article_created_time(上次点击文章的的创建时间), user_clicked_article_created_time_max(浏览过最新的文章的创建时间是什么)
     user_click_last_article_click_time(上次点击文章的时间), user_clicked_article_click_time_mean(点击文章的平均时间戳)
     user_last_click_created_at_ts_diff(当前文章的创建时间与上次点击文章创建时间之差), user_last_click_timestamp_diff(当前文章的创建时间与上次点击文章的时间之差), user_last_click_words_count_diff(当前文章的字数与上次点击文章的字数之差)
     user_id_cnt(用户的活跃信息), article_id_cnt(文章的热门信息), user_id_category_id_cnt(用户对某个类别的兴趣)
    """
    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')

    # 召回相关特征
    ## itemcf 相关
    user_item_ = df_click.groupby('user_id')['article_id'].agg(
        list).reset_index()
    
    user_item_dict = dict(zip(user_item_['user_id'], user_item_['article_id']))

    if mode == 'valid':
        f = open('/home/wangtiantian/shikainan/newscommemder/top2wk/user_data/sim/offline/itemcf_sim.pkl', 'rb')
        item_sim = pickle.load(f)
        f.close()
    else:
        f = open('/home/wangtiantian/shikainan/newscommemder/top2wk/user_data/sim/online/itemcf_sim.pkl', 'rb')
        item_sim = pickle.load(f)
        f.close()

    # 用户历史点击物品与待预测物品相似度
    """
    axis=0 (默认值): 表示逐列操作。函数会依次作用于DataFrame的每一列。
    axis=1: 表示逐行操作。Pandas会遍历DataFrame的每一行，并将每一行作为一个Series对象（你可以把它想象成一个带标签的字典）传递给指定的函数。这正是我们这里需要的。
    """
    df_feature['user_clicked_article_itemcf_sim_sum'] = df_feature[[
        'user_id', 'article_id'
    ]].parallel_apply(func_if_sum, axis=1)
    df_feature['user_last_click_article_itemcf_sim'] = df_feature[[
        'user_id', 'article_id'
    ]].parallel_apply(func_if_last, axis=1)

    """
     df_feature: [user_id, article_id, label, sim_score, category_id, created_at_ts, words_count, created_at_datetime,  
     user_id_click_article_created_at_ts_diff_mean(用户点击文章创建时间平均值), user_id_click_diff_mean(续点击的时间间隔),
     user_click_timestamp_created_at_ts_diff_mean(平均新鲜度), user_click_timestamp_created_at_ts_diff_std(新鲜度稳定性)
     user_click_datetime_hour_std(点击的标准差),  user_clicked_article_words_count_mean(长期阅读文本的篇幅), user_click_last_article_words_count(上次阅读文本的篇幅)
     user_click_last_article_created_time(上次点击文章的的创建时间), user_clicked_article_created_time_max(浏览过最新的文章的创建时间是什么)
     user_click_last_article_click_time(上次点击文章的时间), user_clicked_article_click_time_mean(点击文章的平均时间戳)
     user_last_click_created_at_ts_diff(当前文章的创建时间与上次点击文章创建时间之差), user_last_click_timestamp_diff(当前文章的创建时间与上次点击文章的时间之差), user_last_click_words_count_diff(当前文章的字数与上次点击文章的字数之差)
     user_id_cnt(用户的活跃信息), article_id_cnt(文章的热门信息), user_id_category_id_cnt(用户对某个类别的兴趣)
     user_clicked_article_itemcf_sim_sum(当前文章与用户过去看的文章的相似度之和), user_last_click_article_itemcf_sim(当前文章与用户上次看的文章的相似度)
    """

    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')

    ## binetwork 相关
    # if mode == 'valid':
    #     f = open('/home/wangtiantian/shikainan/newscommemder/top2wk/user_data/sim/offline/binetwork_sim.pkl', 'rb')
    #     binetwork_sim = pickle.load(f)
    #     f.close()
    # else:
    #     f = open('/home/wangtiantian/shikainan/newscommemder/top2wk/user_data/sim/online/binetwork_sim.pkl', 'rb')
    #     binetwork_sim = pickle.load(f)
    #     f.close()

    # df_feature['user_last_click_article_binetwork_sim'] = df_feature[[
    #     'user_id', 'article_id'
    # ]].parallel_apply(func_binetwork_sim_last, axis=1)

    # log.debug(f'df_feature.shape: {df_feature.shape}')
    # log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')

    ## w2v 相关
    if mode == 'valid':
        f = open('/home/wangtiantian/shikainan/newscommemder/top2wk/user_data/data/offline/article_w2v.pkl', 'rb')
        article_vec_map = pickle.load(f)
        f.close()
    else:
        f = open('/home/wangtiantian/shikainan/newscommemder/top2wk/user_data/data/online/article_w2v.pkl', 'rb')
        article_vec_map = pickle.load(f)
        f.close()


    df_feature['user_last_click_article_w2v_sim'] = df_feature[[
        'user_id', 'article_id'
    ]].parallel_apply(func_w2w_last_sim, axis=1)
    df_feature['user_click_article_w2w_sim_sum_2'] = df_feature[[
        'user_id', 'article_id'
    ]].parallel_apply(lambda x: func_w2w_sum(x, 2), axis=1)

    """
     df_feature: [user_id, article_id, label, sim_score, category_id, created_at_ts, words_count, created_at_datetime,  
     user_id_click_article_created_at_ts_diff_mean(用户点击文章创建时间平均值), user_id_click_diff_mean(续点击的时间间隔),
     user_click_timestamp_created_at_ts_diff_mean(平均新鲜度), user_click_timestamp_created_at_ts_diff_std(新鲜度稳定性)
     user_click_datetime_hour_std(点击的标准差),  user_clicked_article_words_count_mean(长期阅读文本的篇幅), user_click_last_article_words_count(上次阅读文本的篇幅)
     user_click_last_article_created_time(上次点击文章的的创建时间), user_clicked_article_created_time_max(浏览过最新的文章的创建时间是什么)
     user_click_last_article_click_time(上次点击文章的时间), user_clicked_article_click_time_mean(点击文章的平均时间戳)
     user_last_click_created_at_ts_diff(当前文章的创建时间与上次点击文章创建时间之差), user_last_click_timestamp_diff(当前文章的创建时间与上次点击文章的时间之差), user_last_click_words_count_diff(当前文章的字数与上次点击文章的字数之差)
     user_id_cnt(用户的活跃信息), article_id_cnt(文章的热门信息), user_id_category_id_cnt(用户对某个类别的兴趣)
     user_clicked_article_itemcf_sim_sum(当前文章与用户过去看的文章所有的itemcf相似度之和), user_last_click_article_itemcf_sim(当前文章与用户上次看的文章的相似度)
     user_last_click_article_w2v_sim(当前文章与用户过去看的文章所有的emebdding相似度之和), user_click_article_w2w_sim_sum_2(当前文章与用户上次看的2篇文章的emebdding相似度之和)
    """

    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')

    # 保存特征文件
    if mode == 'valid':
        df_feature.to_pickle('/home/wangtiantian/shikainan/newscommemder/top2wk/user_data/data/offline/feature.pkl')

    else:
        df_feature.to_pickle('/home/wangtiantian/shikainan/newscommemder/top2wk/user_data/data/online/feature.pkl')
