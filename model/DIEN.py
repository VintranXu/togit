import torch
from torch import nn 
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class AUGRU(nn.Module):
    """注意力更新门GRU单元"""
    def __init__(self, input_size, hidden_size):
        super(AUGRU, self).__init__()
        self.hidden_size = hidden_size
        
        # 更新门
        self.W_ir = nn.Linear(input_size, hidden_size, bias=False)
        self.W_hr = nn.Linear(hidden_size, hidden_size, bias=False)
        self.b_r = nn.Parameter(torch.zeros(hidden_size))
        
        # 候选隐状态
        self.W_ih = nn.Linear(input_size, hidden_size, bias=False)
        self.W_hh = nn.Linear(hidden_size, hidden_size, bias=False)
        self.b_h = nn.Parameter(torch.zeros(hidden_size))
        
    def forward(self, inputs, hidden, att_score):
        """
        inputs: (batch, input_size)
        hidden: (batch, hidden_size)
        att_score: (batch, 1) 注意力分数
        """
        # 更新门 (引入注意力)
        
        u = torch.sigmoid(self.W_ir(inputs) + self.W_hr(hidden) + self.b_r)
        u = att_score * u  # 注意力加权
        
        # 候选隐状态
        h_tilde = torch.tanh(self.W_ih(inputs) + self.W_hh(hidden * u) + self.b_h)
        
        # 最终隐状态
        hidden = (1 - u) * hidden + u * h_tilde
        
        return hidden


class InterestExtractor(nn.Module):
    """兴趣抽取层（使用GRU）"""
    def __init__(self, input_size, hidden_size):
        super(InterestExtractor, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True, bidirectional=False)
        
    def forward(self, inputs, lengths):
        # Pack序列
        """
        inputs: (batch_size, seq_len, input_size) 的padded序列
        lengths: (batch_size,) 每个样本的实际长度
        batch_first=True: 批次维度在第0维
        enforce_sorted=False: 不要求按长度排序
        """
        packed = pack_padded_sequence(inputs, lengths, batch_first=True, enforce_sorted=False)
        output, hidden = self.gru(packed)  # (num_layers, batch_size, hidden_size)

        """
        返回的output
        形状：(batch_size, max_seq_len, hidden_size)
        含义：每个用户在每个时间步的兴趣状态
        用途：后续兴趣演化层会使用这个完整的兴趣序列

        返回的hidden
        形状：(batch_size, hidden_size)
        含义：每个用户的最终兴趣状态
        用途：可以作为用户的整体兴趣表示
        """
        # output, hidden.squeeze(0)
        output, _ = pad_packed_sequence(output, batch_first=True)
        
        return output, hidden.squeeze(0)


class InterestEvolver(nn.Module):
    """兴趣演化层（使用AUGRU）"""
    def __init__(self, input_size, target_dim, hidden_size):
        super(InterestEvolver, self).__init__()
        self.hidden_size = hidden_size

        att_input_dim = input_size + target_dim
        
        # 注意力网络
        self.att_layer1 = nn.Linear(att_input_dim, 80)
        self.att_layer2 = nn.Linear(80, 40)
        self.att_layer3 = nn.Linear(40, 1)
        
        # AUGRU单元
        self.augru = AUGRU(input_size, hidden_size)
        
    def forward(self, interests, target_embedding, lengths):
        """
        interests: (batch, seq_len, hidden_size) 兴趣序列
        target_embedding: (batch, hidden_size) 目标物品嵌入
        lengths: (batch,) 序列长度
        """
        batch_size, seq_len, hidden_size = interests.shape
        
        # 计算注意力分数
        """
        .expand(-1, seq_len, -1)

        这是问题的核心。.expand() 接受你期望的新张量的形状作为参数。
        它会将原始张量中大小为 1 的维度扩展到参数中指定的大小。
        对于大小不为 1 的维度，参数中指定的数字必须与原始大小相同，或者使用 -1。
        """

        #(batch_size, hidden_size) -> (batch_size, 1, hiddend_size) -> (batch_size, seq_len, hidden_size)
        target_repeat = target_embedding.unsqueeze(1).expand(-1, seq_len, -1)  

        #att_input = (batch， seq_len, hidden_size + hidden_size) = (batch， seq_len, hidden_size * 2)
        att_input = torch.cat([interests, target_repeat], dim=-1)
        
        att_hidden = torch.relu(self.att_layer1(att_input))      # (batch_size, seq_len, 80)
        att_hidden = torch.relu(self.att_layer2(att_hidden))     # (batch_size, seq_len, 40)
        att_scores = torch.sigmoid(self.att_layer3(att_hidden))  # (batch, seq_len, 1) 表示用户交互过的每一个商品与当前商品之间的相似度
        
        # 逐步演化兴趣
        hidden = torch.zeros(batch_size, hidden_size).to(interests.device)
        
        """
        当前的兴趣(hidden) = 当前的历史行为 + 上一次的兴趣状态 + 注意力
        
        """
        for t in range(seq_len):
            hidden = self.augru(interests[:, t, :], hidden, att_scores[:, t, :])
        
        return hidden


class DIEN(nn.Module):
    """DIEN模型主体"""
    def __init__(self, n_users, n_items, n_categories, embedding_dim=128, 
                 hidden_dim=128, num_features=10):
        super(DIEN, self).__init__()
        
        # Embedding层
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        self.category_embedding = nn.Embedding(n_categories, embedding_dim // 2)
        

        self.hist_embedding_dim = embedding_dim + embedding_dim // 2  # item_emb + cat_emb
        self.target_embedding_dim = embedding_dim + embedding_dim // 2  # item_emb + cat_emb

        # 兴趣抽取层
        self.interest_extractor = InterestExtractor(
            self.hist_embedding_dim,  # item_emb + cat_emb
            hidden_dim
        )
        
        # 兴趣演化层
        self.interest_evolver = InterestEvolver(hidden_dim, self.target_embedding_dim, hidden_dim)
        
        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_dim * 2 + embedding_dim * 2 + embedding_dim // 2 + num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)
                
    def forward(self, user_ids, article_ids, category_ids, 
                hist_items, hist_cats, seq_lens, features):
        """
        前向传播
        """
        batch_size = user_ids.shape[0]
        
        # 获取嵌入
        """
        user_emb: (batch_size, embedding_dim)

        target_item_emb: (batch_size, embedding_dim)

        target_cat_emb: (batch_size, embedding_dim / 2)
        """
        user_emb = self.user_embedding(user_ids)
        target_item_emb = self.item_embedding(article_ids)
        target_cat_emb = self.category_embedding(category_ids)
        
        # 历史序列嵌入
        hist_item_emb = self.item_embedding(hist_items)                             #hist_item_emb: (batch_size, seq_len, embedding_dim)
        hist_cat_emb = self.category_embedding(hist_cats)                           #hist_cat_emb: (batch_size, seq_len, embedding_dim // 2)
        hist_emb = torch.cat([hist_item_emb, hist_cat_emb], dim=-1)                 #hist_emb: (batch_size, seq_len, embedding_dim + embedding_dim // 2)
        
        # 兴趣抽取
        valid_lens = seq_lens.cpu().numpy()
        valid_lens[valid_lens == 0] = 1  # 避免0长度
        interests, _ = self.interest_extractor(hist_emb, valid_lens)                #interests: (batch_size, seq_len, hidden_size)
        
        # 兴趣演化
        evolved_interest = self.interest_evolver(
            interests, 
            torch.cat([target_item_emb, target_cat_emb], dim=-1),
            valid_lens
        )                                                                           # volved_interest:(batch_size, hidden_size), 用户基于历史行为序列，针对当前目标物品动态演化出的个性化兴趣表示。
        
        # 特征拼接
        concat_features = torch.cat([
            user_emb,
            target_item_emb,
            target_cat_emb,
            evolved_interest,
            interests[:, -1, :],  # 最后一个兴趣状态
            features
        ], dim=-1)
        
        # 输出预测
        output = self.fc_layers(concat_features)
        return torch.sigmoid(output)   #输出当前的概率