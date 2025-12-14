import torch
from torch import nn as nn

class YouTubeDNN(nn.Module):
    """
    YouTube DNN候选生成模型
    
    结构对应图中:
    - 底层: embedded video watches (历史观看视频embedding) -> watch vector (平均池化)
    - 底层: 可选的其他特征 (如: geographic embedding, example age, gender等)
    - 中间层: 多层ReLU全连接网络
    - 顶层: softmax分类器 (训练阶段) / 最近邻索引 (serving阶段)
    """

    def __init__(self, user_num, item_num, embedding_dim = 64, hidden_units =[512, 256, 128]):
        """
        初始化YouTube DNN模型
        
        参数:
            user_num (int): 用户总数,用于创建用户ID的embedding查找表
            item_num (int): 物品(视频)总数,用于创建物品ID的embedding查找表
            embedding_dim (int): embedding维度,默认64
                                对应图中每个embedding向量的维度
            hidden_units (list): DNN隐藏层单元数列表,默认[512, 256, 128]
                                对应图中的多层ReLU网络
        """
        super(YouTubeDNN, self).__init__()

        self.user_num = user_num  # 用户总数
        self.item_num = item_num  # 物品(视频)总数
        self.embedding_dim = embedding_dim  # embedding维度
        self.hidden_units = hidden_units  # 隐藏层单元数
        
        # ========== Embedding层 ==========
        # ★★★ 注意: 这是简化版本的YouTube DNN ★★★
        # 原始论文中包含的特征:
        #   1. watch vector: 历史观看视频的平均embedding ✓ (已实现)
        #   2. search vector: 搜索词的平均embedding ✗ (未实现)
        #   3. geographic embedding: 地理位置embedding ✗ (未实现)
        #   4. example age: 视频年龄(距离上传时间) ✗ (未实现)
        #   5. gender: 用户性别 ✗ (未实现)
        # 当前实现只使用了: user_id embedding + watch vector
        
        # 用户ID的embedding层 (简化特征,代替gender等人口统计学特征)
        # Shape: (user_num+1, embedding_dim)
        self.user_embedding = nn.Embedding(user_num + 1, embedding_dim, padding_idx=0)
        
        # 物品(视频)ID的embedding层
        # 对应图中底层的"embedded video watches"
        # Shape: (item_num+1, embedding_dim)
        self.item_embedding = nn.Embedding(item_num + 1, embedding_dim, padding_idx=0)


        # ========== DNN网络 (对应图中的多层ReLU) ==========
        # 输入维度 = user_emb + hist_emb (watch vector)
        # 对应图中将watch vector和user vector拼接后输入到ReLU层
        # 注意: 如果添加更多特征(geographic, age, gender等),需要增加input_dim
        input_dim = self.embedding_dim * 2  # user_emb维度 + hist_emb维度
        # 例: embedding_dim=64时, input_dim=128
        # 完整版可能是: embedding_dim * 2 + geo_dim + age_dim + gender_dim + ...

        layers = []

        # 构建多层全连接网络,对应图中的多层ReLU
        for i, units in enumerate(hidden_units):
            if i == 0:
                # 第一层: input_dim -> hidden_units[0]
                # 例: 128 -> 512
                layers.append(nn.Linear(input_dim, units))
            else:
                # 后续层: hidden_units[i-1] -> hidden_units[i]
                # 例: 512 -> 256, 256 -> 128
                layers.append(nn.Linear(hidden_units[i - 1], units))
            
            layers.append(nn.ReLU())  # 对应图中的ReLU激活函数
            layers.append(nn.Dropout(0.4))  # Dropout正则化

        # 最终用户表征层: hidden_units[-1] -> embedding_dim
        # 将最后一层映射回embedding维度,得到最终的user vector u
        # 对应图中顶部的user vector u
        # 例: 128 -> 64
        layers.append(nn.Linear(hidden_units[-1], embedding_dim))

        self.dnn = nn.Sequential(*layers)

        # 权重初始化
        self.__init__weights()

    def __init__weights(self):
        """
        初始化模型的权重
        使用Xavier初始化方法保证训练稳定性
        """
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)


    def forward(self, user_ids, histories, target_items):
        """
        前向传播
        
        参数:
            user_ids: 用户ID
                     Shape: (batch_size,)
            histories: 用户历史观看的视频ID序列
                      对应图中的"embedded video watches"
                      Shape: (batch_size, seq_len)
                      其中seq_len是历史序列长度,padding用0填充
            target_items: 目标物品(候选视频)ID
                         对应图中的"video vectors v_j"
                         Shape: (batch_size,)
        
        返回:
            scores: 用户对目标物品的评分 (user vector u 和 video vector v_j 的点积)
                   Shape: (batch_size,)
            user_final_emb: 最终的用户向量u (对应图中顶部的user vector u)
                           Shape: (batch_size, embedding_dim)
            item_emb: 目标物品向量v_j
                     Shape: (batch_size, embedding_dim)
        """
        # ========== 1. 用户embedding (简化特征) ==========
        # 实际YouTube DNN可能不直接使用user_id,这里作为辅助特征
        user_emb = self.user_embedding(user_ids)  
        # Shape: (batch_size, embedding_dim)
        # 例: (256, 64)

        # ========== 2. 历史行为embedding并平均池化 ==========
        # 对应图中: embedded video watches -> average -> watch vector
        hist_emb = self.item_embedding(histories)  
        # Shape: (batch_size, seq_len, embedding_dim)
        # 例: (256, 50, 64) - 每个用户最多50个历史视频

        # 创建mask,忽略padding的0
        mask = (histories != 0).float().unsqueeze(-1) 
        # Shape: (batch_size, seq_len, 1)
        # 例: (256, 50, 1) - 标记哪些位置是真实视频(1)或padding(0)
        
        hist_emb = hist_emb * mask  
        # Shape: (batch_size, seq_len, embedding_dim)
        # 将padding位置的embedding置零
        
        hist_lengths = mask.sum(dim=1)  
        # Shape: (batch_size, 1)
        # 统计每个用户实际的历史视频数量
        
        hist_lengths = torch.clamp(hist_lengths, min=1)  # 避免除零
        # Shape: (batch_size, 1)

        # 平均池化: 对应图中的"average"操作,得到watch vector
        # 池化方式: 对seq_len维度求和,然后除以实际长度
        # 具体过程:
        #   1. hist_emb.sum(dim=1): 在seq_len维度(dim=1)上求和
        #      - 将每个用户的所有历史视频embedding在对应位置相加
        #      - 例: 如果用户有3个历史视频,每个embedding是64维
        #            embedding_1 = [0.1, 0.2, ..., 0.5]  (64维)
        #            embedding_2 = [0.3, 0.1, ..., 0.2]  (64维)
        #            embedding_3 = [0.2, 0.4, ..., 0.3]  (64维)
        #            sum结果   = [0.6, 0.7, ..., 1.0]  (64维) <- 对应位置相加
        #      - Shape变化: (batch_size, seq_len, embedding_dim) -> (batch_size, embedding_dim)
        #   2. / hist_lengths: 除以实际历史视频数量(排除padding的0)
        #      - 例: sum结果 / 3 = [0.2, 0.233, ..., 0.333]  (64维)
        #      - 得到每个维度的平均值
        hist_emb = hist_emb.sum(dim=1) / hist_lengths 
        # Shape: (batch_size, embedding_dim)
        # 例: (256, 64) - 这就是图中的"watch vector"
        # 每个用户得到一个64维的平均向量,表征其历史观看偏好

        # ========== 3. 目标物品的embedding ==========
        # 对应图中的"video vectors v_j"
        item_emb = self.item_embedding(target_items)  
        # Shape: (batch_size, embedding_dim)
        # 例: (256, 64)

        # ========== 4. 拼接用户表征 ==========
        # 将user embedding和watch vector拼接
        # 对应图中将多个特征向量拼接后输入到ReLU网络
        user_repr = torch.cat([user_emb, hist_emb], dim = 1) 
        # Shape: (batch_size, embedding_dim * 2)
        # 例: (256, 128)

        # ========== 5. 通过DNN网络得到最终用户向量 ==========
        # 对应图中: 输入 -> 多层ReLU -> user vector u
        user_final_emb = self.dnn(user_repr)  
        # Shape: (batch_size, embedding_dim)
        # 例: (256, 64) - 这就是图中顶部的"user vector u"

        # ========== 6. 计算相似性分数(点积) ==========
        # 训练阶段: 对应图中user vector u和video vector v_j的点积
        # 然后通过softmax计算class probabilities
        scores = torch.sum(user_final_emb * item_emb, dim = 1)
        # Shape: (batch_size,)
        # 例: (256,) - 每个样本一个分数

        return scores, user_final_emb, item_emb

    def get_user_embeddings(self, user_ids, histories):
        """
        获取用户的最终embedding向量 (对应图中的user vector u)
        用于serving阶段: 预先计算所有用户的向量,存储到向量数据库
        
        参数:
            user_ids: 用户ID
                     Shape: (batch_size,)
            histories: 用户历史观看的视频ID序列
                      Shape: (batch_size, seq_len)
        
        返回:
            user_final_emb: 最终的用户向量u
                           Shape: (batch_size, embedding_dim)
                           对应图中serving阶段的user vector u
        """
        self.eval()  # 设置为评估模式
        with torch.no_grad():  # 不计算梯度
            # 用户embedding
            user_emb = self.user_embedding(user_ids)
            # Shape: (batch_size, embedding_dim)
            
            # 历史视频embedding
            hist_emb = self.item_embedding(histories)
            # Shape: (batch_size, seq_len, embedding_dim)

            # 创建mask并平均池化
            mask = (histories != 0).float().unsqueeze(-1)
            # Shape: (batch_size, seq_len, 1)
            
            hist_emb = hist_emb * mask
            hist_lens = mask.sum(dim=1)
            hist_lens = torch.clamp(hist_lens, min=1) #避免除0
            
            # 平均池化得到watch vector
            hist_emb = hist_emb.sum(dim = 1) / hist_lens
            # Shape: (batch_size, embedding_dim)

            # 拼接特征
            user_repr = torch.cat([user_emb, hist_emb], dim = 1)
            # Shape: (batch_size, embedding_dim * 2)
            
            # 通过DNN得到最终用户向量u
            user_final_emb = self.dnn(user_repr)
            # Shape: (batch_size, embedding_dim)
            
            return user_final_emb
    
    def get_item_embeddings(self, item_ids):
        """
        获取物品(视频)的embedding向量
        用于serving阶段: 预先计算所有视频的向量,存储到最近邻索引
        
        参数:
            item_ids: 物品(视频)ID
                     Shape: (batch_size,) 或 (num_items,)
        
        返回:
            item_embeddings: 物品向量v_j
                            Shape: (batch_size, embedding_dim)
                            对应图中的"video vectors v_j",存储在nearest neighbor index中
        """
        self.eval()  # 设置为评估模式

        with torch.no_grad():  # 不计算梯度
            # 直接返回视频的embedding向量
            # Shape: (batch_size, embedding_dim)
            return self.item_embedding(item_ids)


