
import torch.nn as nn
import torch
from torch.nn.functional import normalize
from utils.transformer import  TransformerEncoderLayer

class Network(nn.Module):
    def __init__(self, backbone, tokenizer, feature_dim, class_num):
        super(Network, self).__init__()
        self.backbone = backbone
        self.tokenizer = tokenizer
        self.feature_dim = feature_dim
        self.cluster_num = class_num
        self.emb_size = self.backbone.config.hidden_size
        self.instance_projector = nn.Sequential(
            nn.Linear(self.emb_size, self.emb_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.emb_size, 128)
        )
        self.cluster_projector = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, self.emb_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(self.emb_size, self.emb_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.emb_size, self.cluster_num)
        )

        self.TransformerEncoderLayer = TransformerEncoderLayer(128, nhead=1, dim_feedforward=256)
        self.Common_view = nn.Sequential(nn.Linear(128 * 2, 128))



    def forward(self, input_ids, attention_mask):
        input_ids_1, input_ids_2 = torch.unbind(input_ids.squeeze(),dim=1)
        attention_mask_1, attention_mask_2 = torch.unbind(attention_mask.squeeze(), dim=1)
        v_i = self.get_mean_embeddings(input_ids_1, attention_mask_1)
        v_j = self.get_mean_embeddings(input_ids_2, attention_mask_2)

        z_i = self.instance_projector(v_i)
        z_j = self.instance_projector(v_j)


        h_hat1, S_wight1 = self.TransformerEncoderLayer(z_i)
        h_hat2, S_wight2 = self.TransformerEncoderLayer(z_j)

        c_i = self.cluster_projector(h_hat1)
        c_j = self.cluster_projector(h_hat2)

        z_i = normalize(z_i, dim=1)
        z_j = normalize(z_j, dim=1)
        h_hat1 = normalize(h_hat1, dim=1)
        h_hat2 = normalize(h_hat2, dim=1)

        return z_i, z_j, c_i, c_j, h_hat1, h_hat2, S_wight1, S_wight2

    def forward_c(self, x):
        h = self.backbone.encode(x, batch_size=len(x),
                                 convert_to_numpy=False,
                                 convert_to_tensor=True)
        c = self.cluster_projector(h)
        c = torch.nn.functional.softmax(c, dim=1)
        return c

    def forward_c_psd(self, x_j, pseudo_index):
        x = []
        size = len(x_j)
        for i in range(size):
            if pseudo_index[i]:
                x.append(x_j[i])
        h = self.backbone.encode(x, batch_size=len(x),
                                 convert_to_numpy=False,
                                 convert_to_tensor=True)
        c = self.cluster_projector(h)
        c = torch.nn.functional.softmax(c, dim=1)
        return c

    def forward_cluster(self, input_ids, attention_mask):
        v_i = self.get_mean_embeddings(input_ids.squeeze(), attention_mask.squeeze())
        z_i = self.instance_projector(v_i)

        h_hat, S_weight = self.TransformerEncoderLayer(z_i)

        MLP_p = self.cluster_projector(h_hat)
        MLP_label = torch.argmax(MLP_p, dim=1)
        MLP_p = torch.softmax(MLP_p, dim=1)

        return v_i, z_i, MLP_label, MLP_p, S_weight

    def get_mean_embeddings(self, input_ids, attention_mask):
        input_ids = input_ids.to('cuda')
        attention_mask = attention_mask.to('cuda')
        bert_output = self.backbone.forward(input_ids=input_ids, attention_mask=attention_mask)
        attention_mask = attention_mask.unsqueeze(-1)
        mean_output = torch.sum(bert_output[0]*attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
        return mean_output
