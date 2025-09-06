import torch
from torch import nn as nn

class YouTubeDNN(nn.Module):
    """YouTube DNN候选生成模型"""

    def __init__(self, user_num, item_num, embedding_dim = 64, hidden_units =[512, 256, 128]):
        super(YouTubeDNN, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.embedding_dim = embedding_dim
        self.hidden_units = hidden_units
        
        #Embedding层
        self.user_embedding = nn.Embedding(user_num + 1, embedding_dim, padding_idx=0)
        self.item_embedding = nn.Embedding(item_num + 1, embedding_dim, padding_idx=0)


        #DIN 网络
        input_dim = self.embedding_dim * 2  # user_emb + hist_emb

        layers = []

        for i, units in enumerate(hidden_units):
            if i == 0:
                layers.append(nn.Linear(input_dim, units))
            else:
                layers.append(nn.Linear(hidden_units[i - 1], units))
            
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.4))

        #最终用户表征层
        layers.append(nn.Linear(hidden_units[-1], embedding_dim))

        self.dnn = nn.Sequential(*layers)

        #权重初始化
        self.__init__weights()

    def __init__weights(self):
        #初始化模型的权重
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)


    def forward(self, user_ids, histories, target_items):
        #用户embedding
        user_emb = self.user_embedding(user_ids)  #(batch_size, embedding_dim)

        #历史行为embedding并平均池化
        hist_emb = self.item_embedding(histories)  #(batch_size, seq_len, embedding_dim)

        #创建mask，忽略padding的0
        mask = (histories != 0).float().unsqueeze(-1) #(batch_size, seq_len, 1)
        hist_emb = hist_emb * mask  #(batch_size, seq_len, embedding_dim)
        hist_lengths = mask.sum(dim=1)  #(batch_size, 1)
        hist_lengths = torch.clamp(hist_lengths, min=1)  # 避免除零  #(batch_size, 1)

        hist_emb = hist_emb.sum(dim=1) / hist_lengths #[batch_size, embedding_dim]

        #目标物品的embedding
        item_emb = self.item_embedding(target_items)  #(batch_size, embedding_dim)

        #拼接用户表征
        user_repr = torch.cat([user_emb, hist_emb], dim = 1) #(batch_size, embedding_dim * 2)

        #DNN网络
        user_final_emb = self.dnn(user_repr)  #(batch_size, embedding_dim)

        #计算相似性分数(点积)
        scores = torch.sum(user_final_emb * item_emb, dim = 1)

        return scores, user_final_emb, item_emb

    def get_user_embeddings(self, user_ids, histories):
        #获取用户的embedding向量
        self.eval()
        with torch.no_grad():
            user_emb = self.user_embedding(user_ids)
            hist_emb = self.item_embedding(histories)

            mask = (histories != 0).float().unsqueeze(-1)
            hist_emb = hist_emb * mask
            hist_lens = mask.sum(dim=1)
            hist_lens = torch.clamp(hist_lens, min=1) #避免除0
            
            hist_emb = hist_emb.sum(dim = 1) / hist_lens

            user_repr = torch.cat([user_emb, hist_emb], dim = 1)  #(B, D * 2)
            user_final_emb = self.dnn(user_repr) #(B, D)
            return user_final_emb
    
    def get_item_embeddings(self, item_ids):
        self.eval()

        with torch.no_grad():
            return self.item_embedding(item_ids)


