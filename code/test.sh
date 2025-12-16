time=$(date "+%Y-%m-%d-%H:%M:%S")

# 处理数据
python code/data.py --mode valid --logfile "${time}.log"

# itemcf 召回
python code/recall_itemcf.py --mode valid --logfile "${time}.log"

# binetwork 召回
python code/recall_binetwork.py --mode valid --logfile "${time}.log"

# w2v 召回
python code/recall_w2v.py --mode valid --logfile "${time}.log"

# 召回合并
python code/recall.py --mode valid --logfile "${time}.log"

# 排序特征
python code/rank_feature.py --mode valid --logfile "${time}.log"

# lgb 模型训练
python code/rank_lgb.py --mode valid --logfile "${time}.log"


# valid 用于离线验证，online 用于生产环境