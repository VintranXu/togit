import argparse
import gc
import os
import random
import warnings

import joblib
import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from utils import Logger, evaluate, gen_sub

warnings.filterwarnings('ignore')

seed = 2020
random.seed(seed)

# 命令行参数
parser = argparse.ArgumentParser(description='lightgbm 排序')
parser.add_argument('--mode', default='predict_only')
parser.add_argument('--logfile', default='test.log')

args = parser.parse_args()

mode = args.mode
logfile = args.logfile

# 初始化日志
os.makedirs('/home/wangtiantian/shikainan/newscommemder/top2wk/user_data/log', exist_ok=True)
log = Logger(f'/home/wangtiantian/shikainan/newscommemder/top2wk/user_data/log/{logfile}').logger
log.info(f'lightgbm 排序，mode: {mode}')


def train_model(df_feature, df_query):
    # 1.根据label是否为空，切分训练集和测试集
    df_train = df_feature[df_feature['label'].notnull()]
    df_test = df_feature[df_feature['label'].isnull()]
    
    # 2. 及时释放不再使用的DataFrame，节约内存
    del df_feature
    gc.collect()


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
    ycol = 'label'
    feature_names = list(
        filter(
            lambda x: x not in [ycol, 'created_at_datetime', 'click_datetime'],
            df_train.columns))
    
    print(feature_names)
    # 对每一列的列名进行排序
    feature_names.sort()

    # 初始化一个LightGBM分类器，并配置超参数。
    model = lgb.LGBMClassifier(num_leaves=64,
                               max_depth=10,
                               learning_rate=0.05,
                               n_estimators=10000,
                               subsample=0.8,
                               feature_fraction=0.8,
                               reg_alpha=0.5,
                               reg_lambda=0.5,
                               random_state=seed,
                               importance_type='gain',
                               metric=None)
    
    # 这个列表将用来收集每一折交叉验证中对验证集
    oof = []
    # 创建一个新的DataFrame prediction，它只包含测试集用户的 user_id 和 article_id，作为存储最终预测结果的“骨架”。
    prediction = df_test[['user_id', 'article_id']]
    # DataFrame 中创建一个名为 pred 的新列，并将其初始值全部设为0。这个列将用来累加5个模型对测试集的预测概率。
    prediction['pred'] = 0
    # 初始化一个空列表，用来收集每一折训练出的模型的特征重要性。
    df_importance_list = []

    # 训练模型
    kfold = GroupKFold(n_splits=5)

    """
    df_train[feature_names]: 特征数据 (X)。
    df_train[ycol]: 标签数据 (y)。
    groups=df_train['user_id']: 这是最重要的参数！ 我们在这里把 user_id 这一列传给了 groups 参数。这就等于告诉 GroupKFold：“请把 user_id 相同的所有行都当作同一个组来处理”。
    """
    for fold_id, (trn_idx, val_idx) in enumerate(
            kfold.split(df_train[feature_names], df_train[ycol],
                        df_train['user_id'])):
        # 使用 .iloc 和训练集索引 trn_idx，选出训练数据行，再从中选出特征列
        X_train = df_train.iloc[trn_idx][feature_names]
        # 使用 .iloc 和训练集索引 trn_idx，选出训练数据行，再从中选出标签列
        Y_train = df_train.iloc[trn_idx][ycol]

        X_val = df_train.iloc[val_idx][feature_names]
        Y_val = df_train.iloc[val_idx][ycol]

        log.debug(
            f'\nFold_{fold_id + 1} Training ================================\n'
        )

        """
        X_train: 当前这一折的训练集特征。这是模型的输入（问题）。
        Y_train: 当前这一折的训练集标签。这是模型需要学习的目标（答案）。
        eval_names: 为监控集（eval_set）中的数据集命名。
        eval_set: 提供监控集（评估集），这是实现早停机制和观察模型性能的关键。
        verbose: 控制训练过程中日志的打印频率。
        eval_metric: 指定在eval_set上使用哪一个评估指标来衡量模型性能。
        """
        lgb_model = model.fit(X_train,
                              Y_train,
                              eval_names=['train', 'valid'],
                              eval_set=[(X_train, Y_train), (X_val, Y_val)],
                              callbacks=[
                                    lgb.log_evaluation(period=100),
                                    lgb.early_stopping(stopping_rounds=100)
                                ],
                              eval_metric='auc',
                              )

        # 1. 使用训练好的模型，对当前折的验证集 X_val 进行预测
        pred_val = lgb_model.predict_proba(
            X_val, num_iteration=lgb_model.best_iteration_)[:, 1]
        
        # 2. 创建一个包含验证集身份信息和真实标签的DataFrame
        df_oof = df_train.iloc[val_idx][['user_id', 'article_id', ycol]].copy()
        df_oof['pred'] = pred_val
        oof.append(df_oof)


        # 1. 使用当前折的模型，对完整的测试集 df_test 进行预测
        pred_test = lgb_model.predict_proba(
            df_test[feature_names], num_iteration=lgb_model.best_iteration_)[:,1]
        
        # 2. 将预测出的概率除以5，然后累加到 prediction['pred'] 列上                                                              1]
        prediction['pred'] += pred_test / 5

        # 1. 创建一个包含特征名和其重要性分数的DataFrame
        df_importance = pd.DataFrame({
            'feature_name':
            feature_names,
            'importance':
            lgb_model.feature_importances_,
        })
        df_importance_list.append(df_importance)

        joblib.dump(model, f'/home/wangtiantian/shikainan/newscommemder/top2wk/user_data/model/lgb{fold_id}.pkl')

    # 特征重要性
    df_importance = pd.concat(df_importance_list)
    df_importance = df_importance.groupby([
        'feature_name'
    ])['importance'].agg('mean').sort_values(ascending=False).reset_index()
    log.debug(f'importance: {df_importance}')

    # 生成线下
    df_oof = pd.concat(oof)
    df_oof.sort_values(['user_id', 'pred'],
                       inplace=True,
                       ascending=[True, False])
    log.debug(f'df_oof.head: {df_oof.head()}')

    # 计算相关指标
    total = df_query[df_query['click_article_id'] != -1].user_id.nunique()
    hitrate_5, mrr_5, hitrate_10, mrr_10, hitrate_20, mrr_20, hitrate_40, mrr_40, hitrate_50, mrr_50 = evaluate(
        df_oof, total)
    log.debug(
        f'{hitrate_5}, {mrr_5}, {hitrate_10}, {mrr_10}, {hitrate_20}, {mrr_20}, {hitrate_40}, {mrr_40}, {hitrate_50}, {mrr_50}'
    )

    # 生成提交文件
    df_sub = gen_sub(prediction)
    df_sub.sort_values(['user_id'], inplace=True)
    os.makedirs('/home/wangtiantian/shikainan/newscommemder/top2wk/prediction_result', exist_ok=True)
    df_sub.to_csv(f'/home/wangtiantian/shikainan/newscommemder/top2wk/prediction_result/result.csv', index=False)

def predict_with_saved_models(df_feature, df_query=None):
    """
    使用已保存的模型进行预测
    
    Args:
        df_feature: 特征数据
        df_query: 查询数据（用于评估，可选）
    """
    log.info("使用已保存的模型进行预测...")
    
    # 根据label是否为空，切分训练集和测试集
    if 'label' in df_feature.columns:
        df_train = df_feature[df_feature['label'].notnull()]
        df_test = df_feature[df_feature['label'].isnull()]
    else:
        df_test = df_feature
        df_train = None
    
    ycol = 'label'
    feature_names = list(
        filter(
            lambda x: x not in [ycol, 'created_at_datetime', 'click_datetime'],
            df_test.columns))
    feature_names.sort()
    
    # 检查模型文件是否存在
    model_files_exist = True
    for fold_id in range(5):
        model_path = f'/home/wangtiantian/shikainan/newscommemder/top2wk/user_data/model/lgb{fold_id}.pkl'
        if not os.path.exists(model_path):
            log.error(f"模型文件不存在: {model_path}")
            model_files_exist = False
            
    if not model_files_exist:
        raise FileNotFoundError("部分或全部模型文件不存在，请先训练模型")
    
    # 创建预测DataFrame
    prediction = df_test[['user_id', 'article_id']].copy()
    prediction['pred'] = 0
    
    # 加载模型并预测
    for fold_id in tqdm(range(5), desc="加载模型并预测"):
        model_path = f'/home/wangtiantian/shikainan/newscommemder/top2wk/user_data/model/lgb{fold_id}.pkl'
        log.info(f"加载模型: {model_path}")
        
        model = joblib.load(model_path)
        
        # 预测
        pred_test = model.predict_proba(df_test[feature_names])[:, 1]
        prediction['pred'] += pred_test / 5
        
        log.info(f"Fold {fold_id} 预测完成")
    
    # 如果有训练集标签，计算验证集上的指标
    if df_train is not None and df_query is not None:
        # 对训练集也进行预测（用于评估）
        oof_predictions = []
        
        for fold_id in range(5):
            model = joblib.load(f'/home/wangtiantian/shikainan/newscommemder/top2wk/user_data/model/lgb{fold_id}.pkl')
            pred_train = model.predict_proba(df_train[feature_names])[:, 1]
            
            df_oof = df_train[['user_id', 'article_id', ycol]].copy()
            df_oof['pred'] = pred_train
            oof_predictions.append(df_oof)
        
        # 合并OOF预测
        df_oof = pd.concat(oof_predictions).groupby(['user_id', 'article_id', ycol])['pred'].mean().reset_index()
        df_oof.sort_values(['user_id', 'pred'], inplace=True, ascending=[True, False])
        
        # 计算指标
        total = df_query[df_query['click_article_id'] != -1].user_id.nunique()
        hitrate_5, mrr_5, hitrate_10, mrr_10, hitrate_20, mrr_20, hitrate_40, mrr_40, hitrate_50, mrr_50 = evaluate(
            df_oof, total)
        log.info(
            f'验证集指标: HR@5={hitrate_5:.4f}, MRR@5={mrr_5:.4f}, '
            f'HR@10={hitrate_10:.4f}, MRR@10={mrr_10:.4f}, '
            f'HR@20={hitrate_20:.4f}, MRR@20={mrr_20:.4f}'
        )
    
    # 生成提交文件
    log.info("生成提交文件...")
    df_sub = gen_sub(prediction)
    df_sub.sort_values(['user_id'], inplace=True)
    
    os.makedirs('/home/wangtiantian/shikainan/newscommemder/top2wk/prediction_result', exist_ok=True)
    output_path = f'/home/wangtiantian/shikainan/newscommemder/top2wk/prediction_result/result_loaded.csv'
    df_sub.to_csv(output_path, index=False)
    log.info(f"预测结果已保存到: {output_path}")
    
    return df_sub


def online_predict(df_test):
    ycol = 'label'
    feature_names = list(
        filter(
            lambda x: x not in [ycol, 'created_at_datetime', 'click_datetime'],
            df_test.columns))
    feature_names.sort()

    prediction = df_test[['user_id', 'article_id']]
    prediction['pred'] = 0

    for fold_id in tqdm(range(5)):
        model = joblib.load(f'/home/wangtiantian/shikainan/newscommemder/top2wk/user_data/model/lgb{fold_id}.pkl')
        pred_test = model.predict_proba(df_test[feature_names])[:, 1]
        prediction['pred'] += pred_test / 5

    # 生成提交文件
    df_sub = gen_sub(prediction)
    df_sub.sort_values(['user_id'], inplace=True)
    os.makedirs('/home/wangtiantian/shikainan/newscommemder/top2wk/prediction_result', exist_ok=True)
    df_sub.to_csv(f'/home/wangtiantian/shikainan/newscommemder/top2wk/prediction_result/result.csv', index=False)


if __name__ == '__main__':
    if mode == 'valid':
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
        # 加载特征数据以及查询数据
        df_feature = pd.read_pickle('/home/wangtiantian/shikainan/newscommemder/top2wk/user_data/data/offline/feature.pkl')
        df_query = pd.read_pickle('/home/wangtiantian/shikainan/newscommemder/top2wk/user_data/data/offline/query.pkl')
        # 找出所有包含文本数据的列
        for f in df_feature.select_dtypes('object').columns:
            lbl = LabelEncoder()
            # 它首先将整列的数据强制转换为字符串类型。
            # 最后，将转换后得到的只包含整数的新序列，覆盖掉 df_feature 中原始的那一列。
            df_feature[f] = lbl.fit_transform(df_feature[f].astype(str))

        train_model(df_feature, df_query)

    elif mode == 'predict_only':
        # 纯预测模式，只需要特征文件
        log.info("运行纯预测模式...")
        
        # 自动检测使用哪个特征文件
        offline_feature_path = '/home/wangtiantian/shikainan/newscommemder/top2wk/user_data/data/offline/feature.pkl'
        online_feature_path = '/home/wangtiantian/shikainan/newscommemder/top2wk/user_data/data/online/feature.pkl'
        
        if os.path.exists(offline_feature_path):
            log.info(f"使用离线特征文件: {offline_feature_path}")
            df_feature = pd.read_pickle(offline_feature_path)
            
            # 尝试加载查询数据用于评估
            try:
                df_query = pd.read_pickle('/home/wangtiantian/shikainan/newscommemder/top2wk/user_data/data/offline/query.pkl')
            except:
                df_query = None
                log.warning("无法加载查询数据，跳过评估指标计算")
        elif os.path.exists(online_feature_path):
            log.info(f"使用在线特征文件: {online_feature_path}")
            df_feature = pd.read_pickle(online_feature_path)
            df_query = None
        else:
            raise FileNotFoundError("找不到特征文件")
        
        # 找出所有包含文本数据的列
        for f in df_feature.select_dtypes('object').columns:
            lbl = LabelEncoder()
            df_feature[f] = lbl.fit_transform(df_feature[f].astype(str))
        
        # 使用保存的模型进行预测
        predict_with_saved_models(df_feature, df_query)
    else:
        df_feature = pd.read_pickle('/home/wangtiantian/shikainan/newscommemder/top2wk/user_data/data/online/feature.pkl')
        online_predict(df_feature)
    