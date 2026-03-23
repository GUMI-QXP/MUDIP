# import time
# import shap
# import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import time
from my_utiils import *

import torch
from torch.utils import data
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn
# from torch.utils.tensorboard import SummaryWriter
# from torch.utils.data import SequentialSampler

# from prettytable import PrettyTable
from model_helper import Encoder_MultipleLayers, Embeddings,GeneModel1
# from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from functools import partial
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from dgllife.model.gnn import GCN
skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
from dgl import batch as dgl_batch
import shap

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdDepictor
from rdkit.Chem.rdchem import Mol
import matplotlib.pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
from itertools import combinations
from torch import flatten
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, auc
from scipy import stats

def one_hot_tensor(y, num_dim):
    y_onehot = torch.zeros(y.shape[0], num_dim)
    y_onehot.scatter_(1, y.view(-1,1), 1)
    return y_onehot

class data_process_loader(data.Dataset):
    def __init__(self, list_IDs, labels, drug_df, rna_df,prot_df,cancer_df, max_drug_nodes=290):
        # train_drug.index.values, train_drug.Label.values, train_drug, train_rna
        'Initialization'

        self.list_IDs = list_IDs
        self.labels = labels
        self.drug_df = drug_df
        self.rna_df = rna_df
        self.prot_df = prot_df
        self.cancer_df = cancer_df
        self.max_drug_nodes = max_drug_nodes

        self.atom_featurizer = CanonicalAtomFeaturizer()
        self.bond_featurizer = CanonicalBondFeaturizer(self_loop=True)
        self.fc = partial(smiles_to_bigraph, add_self_loop=True)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        index = self.list_IDs[index]
        v_d = self.drug_df.iloc[index]['smiles']
        v_d = self.fc(smiles=v_d, node_featurizer=self.atom_featurizer, edge_featurizer=self.bond_featurizer)
        actual_node_feats = v_d.ndata.pop('h')
        num_actual_nodes = actual_node_feats.shape[0]
        num_virtual_nodes = self.max_drug_nodes - num_actual_nodes
        virtual_node_bit = torch.zeros([num_actual_nodes, 1])
        actual_node_feats = torch.cat((actual_node_feats, virtual_node_bit), 1)
        v_d.ndata['h'] = actual_node_feats
        if num_virtual_nodes < 0:
            print(f"警告：计算得到的虚拟节点数为负（{num_virtual_nodes}），将其设置为 0")
            print(self.drug_df.iloc[index]['smiles'])
        virtual_node_feat = torch.cat((torch.zeros(num_virtual_nodes, 74), torch.ones(num_virtual_nodes, 1)), 1)
        v_d.add_nodes(num_virtual_nodes, {"h": virtual_node_feat})
        v_d = v_d.add_self_loop()

        identifier = self.drug_df.iloc[index]['identifier']
        v_p = np.array(self.rna_df.iloc[index])  #
        v_pt = np.array(self.prot_df.iloc[index])
        v_c = np.array(self.cancer_df.iloc[index])
        y = self.labels[index]
        # 返回对应的药物特征，rna特征，反应的标签值


        return v_d, v_p, v_pt, v_c,y,identifier


class transformer(nn.Sequential):  # transformer模型提取药物特征
    def __init__(self):
        super(transformer, self).__init__()

        input_dim_drug = 2586
        transformer_emb_size_drug = 64
        transformer_dropout_rate = 0.1
        # transformer_n_layer_drug = 8
        # transformer_intermediate_size_drug = 512
        # transformer_num_attention_heads_drug = 8
        transformer_n_layer_drug = 6
        transformer_intermediate_size_drug = 512
        transformer_num_attention_heads_drug = 4
        transformer_attention_probs_dropout = 0.1
        transformer_hidden_dropout_rate = 0.1

        self.emb = Embeddings(input_dim_drug,
                              transformer_emb_size_drug,
                              50,  # 对应药物特征的50维度
                              transformer_dropout_rate)

        self.encoder = Encoder_MultipleLayers(transformer_n_layer_drug,
                                              transformer_emb_size_drug,
                                              transformer_intermediate_size_drug,
                                              transformer_num_attention_heads_drug,
                                              transformer_attention_probs_dropout,
                                              transformer_hidden_dropout_rate)

    def forward(self, v,epo,state,identifier):
        # 分别接收两个表示药物特征的tensor（64,50）
        e = v[0].long().to(device)
        e_mask = v[1].long().to(device)
        ex_e_mask = e_mask.unsqueeze(1).unsqueeze(2)  # 维度展开->(64,1,1,50)
        ex_e_mask = (1.0 - ex_e_mask) * -10000.0

        emb = self.emb(e)  # 经过emb函数后变为（64,50,128）
        encoded_layers , attention_weights = self.encoder(emb.float(), ex_e_mask.float())  # （64,50,128）

        return encoded_layers[:, 0],attention_weights

class MolecularGCN(nn.Module):
    def __init__(self, in_feats, dim_embedding=128, padding=True, hidden_feats=None, activation=None):
        super(MolecularGCN, self).__init__()
        self.init_transform = nn.Linear(in_feats, dim_embedding, bias=False)
        if padding:
            with torch.no_grad():
                self.init_transform.weight[-1].fill_(0)
        self.gnn = GCN(in_feats=dim_embedding, hidden_feats=hidden_feats, activation=activation)
        self.output_feats = hidden_feats[-1]

    def forward(self, batch_graph):
        node_feats = batch_graph.ndata.pop('h')
        node_feats = self.init_transform(node_feats)
        node_feats = self.gnn(batch_graph, node_feats)
        batch_size = batch_graph.batch_size
        node_feats = node_feats.view(batch_size, -1, self.output_feats)
        v_d_mean = torch.mean(node_feats, dim=1)
        return node_feats

class Attention(nn.Module):
    def __init__(self, input_dim1, input_dim2, hidden_dim):
        super(Attention, self).__init__()

        self.fc1 = nn.Linear(input_dim1, hidden_dim)
        self.fc2 = nn.Linear(input_dim2, hidden_dim)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2):
        # 计算注意力权重
        x1 = self.fc1(x1)
        x2 = self.fc2(x2)
        x = self.relu(x1 + x2)
        attention_weights = self.softmax(self.fc3(x))

        # 对模态进行加权融合
        fused_representation = x1 * attention_weights + x2 * (1 - attention_weights)

        return fused_representation

class Classifier(nn.Sequential):
    def __init__(self, model_drug, model_gene,  model_prot,model_cancer,model_fusion):
    # def __init__(self, model_drug, model_gene,model_gene1,attention):
        super(Classifier, self).__init__()
        self.input_dim_drug = 128
        # self.input_dim_gene = 52
        self.input_dim_gene = 64
        self.input_dim_prot = 64
        self.input_dim_cancer = 64
        self.model_drug = model_drug
        self.model_gene = model_gene
        self.model_prot = model_prot
        self.model_cancer = model_cancer
        self.model_fusion = model_fusion
        # self.model_gene1 = model_gene1
        # self.attention = attention
        self.dropout = nn.Dropout(0.1)
        self.hidden_dims = [256, 128, 64]
        layer_size = len(self.hidden_dims) + 1
        dims = [self.input_dim_drug + self.input_dim_gene+self.input_dim_prot + self.input_dim_cancer] + self.hidden_dims + [2]
        self.predictor = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(layer_size)])
        self.act = nn.Sigmoid()


    def forward(self, v_D, v_P,v_Pt,v_C,label,epo,state,identifier,train_state = True):

        v_D = v_D.to(device)
        v_D = self.model_drug(v_D)#.to(torch.float64)  # v_D->64,128
        v_P=v_P.to(device)
        v_P1 = self.model_gene(v_P)
        v_Pt = v_Pt.to(device)
        v_Pt_1 = self.model_prot(v_Pt)
        v_C = v_C.to(device)
        v_C1 = self.model_cancer(v_C)

        v_fusion_out,inp,FeatureInfo = self.model_fusion(v_P1,v_Pt_1,v_C1)

        v_d_mean = torch.mean(v_D, dim=1)
        v_f = torch.cat((v_d_mean, v_fusion_out), 1)


        for i, l in enumerate(self.predictor):
            if i == (len(self.predictor) - 1):
                v_f = l(v_f)
                v_f = self.act(v_f)

            else:
                v_f = v_f.float()
                # print(v_f.dtype)
                # print(l(v_f).dtype)
                v_f = F.relu(self.dropout(l(v_f)))
        if train_state:
            return v_f,inp,FeatureInfo
        else:
            return v_f


class MLP(nn.Sequential):  # 提取细胞系的基因
    def __init__(self):
        input_dim_gene = 768  #kmer3:998   kmer4:997   kmer5=996   kmer6=995
        hidden_dim_gene = 64
        mlp_hidden_dims_gene = [64, 128]
        super(MLP, self).__init__()
        layer_size = len(mlp_hidden_dims_gene) + 1
        dims = [input_dim_gene] + mlp_hidden_dims_gene + [hidden_dim_gene]
        self.predictor = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(layer_size)])

    def forward(self, v):
        # predict
        v = v.float().to(device)
        for i, l in enumerate(self.predictor):
            # v = F.relu(l(v))
            # v=torch.sigmoid(l(v))
            v=torch.tanh_(l(v))
        return v

class MLP2(nn.Sequential):  # 提取protein的基因
    def __init__(self):
        input_dim_prot = 64
        hidden_dim_prot = 64
        mlp_hidden_dims_prot = [64, 128]
        super(MLP2, self).__init__()
        layer_size = len(mlp_hidden_dims_prot) + 1
        dims = [input_dim_prot] + mlp_hidden_dims_prot + [hidden_dim_prot]
        self.predictor = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(layer_size)])

    def forward(self, v):
        # predict
        v = v.float().to(device)
        for i, l in enumerate(self.predictor):
            # v = F.relu(l(v))
            # v=torch.sigmoid(l(v))
            v=torch.tanh_(l(v))
        return v

class MLP3(nn.Sequential):  # 提取cancer
    def __init__(self):
        input_dim_prot = 89
        hidden_dim_prot = 64
        mlp_hidden_dims_prot = [64, 128]
        super(MLP3, self).__init__()
        layer_size = len(mlp_hidden_dims_prot) + 1
        dims = [input_dim_prot] + mlp_hidden_dims_prot + [hidden_dim_prot]
        self.predictor = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(layer_size)])

    def forward(self, v):
        # predict
        v = v.float().to(device)
        for i, l in enumerate(self.predictor):
            # v = F.relu(l(v))
            # v=torch.sigmoid(l(v))
            v=torch.tanh_(l(v))
        return v


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, weight_decay=0.01):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight_decay = weight_decay

    def forward(self, inputs, targets):
        BCE_loss = nn.BCELoss(reduction='none')(inputs, targets)
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        # 计算权重衰减项
        weight_decay_loss = 0.0
        for param in self.parameters():
            weight_decay_loss += torch.norm(param, p=2)

        loss = focal_loss.mean() + self.weight_decay * weight_decay_loss
        return loss


def xavier_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
           m.bias.data.fill_(0.0)

class LinearLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.clf = nn.Sequential(nn.Linear(in_dim, out_dim))
        self.clf.apply(xavier_init)

    def forward(self, x):
        x = self.clf(x)
        return x


class OvOAttention(nn.Module):
    """
    Module that implements One-vs-Others attention mechanism as proposed by the paper.
    """

    def __init__(self):
        super(OvOAttention, self).__init__()

    def forward(self, others, main, W):
        """
        Compute context vector and attention weights using One-vs-Others attention.

        Args:
            others (List[torch.Tensor]): List of tensors of shape (batch_size, num_heads, seq_len, embed_dim) representing
                                          the other modality inputs.
            main (torch.Tensor): A tensor of shape (batch_size, num_heads, seq_len, embed_dim) representing the main modality input.
            W (torch.nn.Parameter): A learnable parameter tensor of shape (d_head, d_head) representing the weight matrix.

        Returns:
            torch.Tensor: A tensor of shape (batch_size, embed_dim) representing the context vector.
            torch.Tensor: A tensor of shape (batch_size, num_heads, seq_len) representing the attention weights.

        """
        mean = sum(others) / len(others)
        score = mean.squeeze(2) @ W @ main.squeeze(2).transpose(1, 2)
        attn = F.softmax(score, -1)
        context = torch.bmm(attn, main.squeeze(2))
        return context, attn


class MultiHeadAttention(nn.Module):
    """
    Module that implements Multi-Head attention mechanism. This was adapted and modified from https://github.com/sooftware/attentions.

    Args:
        d_model (int): Dimensionality of the input embedding.
        num_heads (int): Number of attention heads.

    Returns:
        torch.Tensor: A tensor of shape (batch_size, seq_len, embed_dim) representing the context vector.
    """

    def __init__(self, d_model: int = 512, num_heads: int = 8):
        super(MultiHeadAttention, self).__init__()

        assert d_model % num_heads == 0, "d_model % num_heads should be zero."

        self.d_head = int(d_model / num_heads)
        self.d_model = d_model
        self.num_heads = num_heads
        self.ovo_attn = OvOAttention()
        self.query_proj = nn.Linear(d_model, self.d_head * num_heads)
        self.key_proj = nn.Linear(d_model, self.d_head * num_heads)
        self.value_proj = nn.Linear(d_model, self.d_head * num_heads)
        self.W = torch.nn.Parameter(torch.FloatTensor(self.d_head, self.d_head).uniform_(-0.1, 0.1))

    def forward(self, other, main):
        """
        Compute context vector using Multi-Head attention.

        Args:
            others (List[torch.Tensor]): List of tensors of shape (batch_size, num_heads, seq_len, embed_dim) representing
                                          the other modality inputs.
            main (torch.Tensor): A tensor of shape (batch_size, num_heads, seq_len, embed_dim) representing the main modality input.

        Returns:
            torch.Tensor: A tensor of shape (batch_size, seq_len, embed_dim) representing the context vector.

        """
        batch_size = main.size(0)
        main = main.unsqueeze(1)
        bsz, tgt_len, embed_dim = main.shape
        src_len, _, _ = main.shape

        main = main.contiguous().view(tgt_len, bsz * self.num_heads, self.d_head).transpose(0, 1)
        main = main.view(bsz, self.num_heads, tgt_len, self.d_head)
        others = []
        for mod in other:
            mod = mod.unsqueeze(1)
            mod = mod.contiguous().view(tgt_len, bsz * self.num_heads, self.d_head).transpose(0, 1)
            mod = mod.view(bsz, self.num_heads, tgt_len, self.d_head)
            others.append(mod)
        context, attn = self.ovo_attn(others, main, self.W)
        context = context.contiguous().view(bsz * tgt_len, embed_dim)
        context = context.view(bsz, tgt_len, context.size(1))

        return context


class MultimodalFramework(nn.Module):
    """
    A PyTorch module for a multimodal framework for the TCGA dataset.

    Args:
    - modality_dims (List[int]): A list of dimensions for each modality. Only tabular modalities are needed.
    - num_heads (int): The number of heads to use in the multihead attention layers.

    """

    def __init__(self, modality_dims, num_heads):
        super().__init__()
        self.num_mod = 3
        self.views = len(modality_dims)
        self.num_heads = num_heads
        self.modality_dims = modality_dims

        # input layers
        self.fc1a = nn.Linear(self.modality_dims[0], 256)
        self.fc2a = nn.Linear(self.modality_dims[1], 256)
        self.fc3a = nn.Linear(self.modality_dims[2], 256)
        # self.fc4a = nn.Linear(self.modality_dims[3], 256)

        ##MLP
        # self.fc2b = nn.Linear(256, 256)
        # self.relu = nn.ReLU()
        # self.dropout2 = nn.Dropout(p=0.2)
        # self.dropout3 = nn.Dropout(p=0.1)
        # gate
        self.FeatureInforEncoder = nn.ModuleList(
            [LinearLayer(modality_dims[view], modality_dims[view]) for view in range(self.views)])
        self.fc2b = nn.Linear(64, 256)
        # attention
        self.pairwise_attention = nn.MultiheadAttention(256, self.num_heads, batch_first=True)
        self.self_attention = nn.MultiheadAttention(256 * self.num_mod, self.num_heads, batch_first=True)
        # self.OvO_attention = MultiHeadAttention(256, self.num_heads)
        self.OvO_attention = MultiHeadAttention(64, self.num_heads)

        # out
        self.out_concat = nn.Linear(256 * self.num_mod, 192)
        # self.out_concat = nn.Linear(192, 64)
        self.out_pairwise = nn.Linear(256 * self.num_mod * (self.num_mod - 1), 192)
        # self.out_OvO = nn.Linear(256 * self.num_mod, 192)
        self.out_OvO = nn.Linear(64 * self.num_mod, 192)

    def bi_directional_att(self, l):

        # All possible pairs in list
        a = list(range(len(l)))
        pairs = list(combinations(a, r=2))
        combos = []
        for pair in pairs:
            # (0,1)
            index_1 = pair[0]
            index_2 = pair[1]
            x = l[index_1]
            y = l[index_2]
            attn_output_LV, attn_output_weights_LV = self.pairwise_attention(x, y, y)
            attn_output_VL, attn_output_weights_VL = self.pairwise_attention(y, x, x)
            combined = torch.cat((attn_output_LV,
                                  attn_output_VL), dim=1)
            combos.append(combined)
        return combos

    def forward(self, inp, model):
        """
        Args:
            - inp: A list of inputs that contains four tensors representing textual features (t1, t2, t3, t4) and one tensor
            representing an image tensor (x).
            - model: A string indicating the type of model to use for the forward pass. It can be "concat", "cross", "self", or "OvO".

        """
        t1, t2, t3 = inp

        t1 = self.fc1a(t1.to(torch.float32))
        t2 = self.fc2a(t2.to(torch.float32))
        t3 = self.fc3a(t3.to(torch.float32))

        outputs = []

        FeatureInfo, feature = dict(), dict()
        for view in range(self.views):
            FeatureInfo[view] = torch.sigmoid(self.FeatureInforEncoder[view](inp[view]))
            feature[view] = inp[view] * FeatureInfo[view]
            # feature[view] = self.fc2b(feature[view])
            outputs.append(feature[view])

        if model == "concat":
            combined = torch.cat(outputs, dim=1)
            # out = self.out_concat(combined)
            out = combined

        elif model == "cross":
            combined = self.bi_directional_att(outputs)
            comb = torch.cat(combined, dim=1)
            out = self.out_pairwise(comb)

        elif model == "self":
            combined = torch.cat(outputs, dim=1)
            comb = self.self_attention(combined, combined, combined)
            out = self.out_concat(comb)

        elif model == "ovo":
            attns = []
            for main in outputs:
                others = list(set(outputs) - set([main]))
                att = self.OvO_attention(others, main)
                attns.append(att.squeeze(1))
            comb = torch.cat(attns, dim=1)
            out = self.out_OvO(comb)

        return out,inp ,FeatureInfo

class GatedFusionThreeModalities(nn.Module):
    """
    Gated Fusion for Three Modalities (x, y, z).
    """

    def __init__(self, input_dim=64, dim=128, output_dim=192, gate_mode='x'):
        """
        Args:
            input_dim (int): 输入模态的维度大小
            dim (int): 隐藏层的中间维度
            output_dim (int): 最终输出类别的数量
            gate_mode (str): 指定用于生成门值的模态，'x', 'y', 'z' 或 'combined'
        """
        super(GatedFusionThreeModalities, self).__init__()

        # 各模态的特征映射层
        self.fc_x = nn.Linear(input_dim, dim)
        self.fc_y = nn.Linear(input_dim, dim)
        self.fc_z = nn.Linear(input_dim, dim)

        # 融合后的输出层
        self.fc_out = nn.Linear(dim, output_dim)

        # 门控机制选择：x, y, z 或组合模式
        self.gate_mode = 'y'
        self.sigmoid = nn.Sigmoid()
        # gate
        self.views = 3
        self.modality_dims =[64,64,64]
        self.FeatureInforEncoder = nn.ModuleList(
            [LinearLayer(self.modality_dims[view], self.modality_dims[view]) for view in range(self.views)])

    def forward(self, x, y, z):
        inp=[x,y,z]
        outputs = []

        FeatureInfo, feature = dict(), dict()
        for view in range(3):
            FeatureInfo[view] = torch.sigmoid(self.FeatureInforEncoder[view](inp[view]))
            feature[view] = inp[view] * FeatureInfo[view]
            # feature[view] = self.fc2b(feature[view])
            outputs.append(feature[view])

        x,y,z=outputs
        # 映射输入模态到隐藏特征空间
        out_x = self.fc_x(x)  # 模态 x 特征
        out_y = self.fc_y(y)  # 模态 y 特征
        out_z = self.fc_z(z)  # 模态 z 特征

        # 根据 gate_mode 生成门值
        if self.gate_mode == 'x':
            gate = self.sigmoid(out_x)  # 使用 x 生成门值
        elif self.gate_mode == 'y':
            gate = self.sigmoid(out_y)  # 使用 y 生成门值
        elif self.gate_mode == 'z':
            gate = self.sigmoid(out_z)  # 使用 z 生成门值
        elif self.gate_mode == 'combined':
            # 使用所有模态特征的组合生成门值
            combined = out_x + out_y + out_z
            gate = self.sigmoid(combined)
        else:
            raise ValueError("Invalid gate_mode. Choose from 'x', 'y', 'z', or 'combined'.")

        # 对三个模态特征进行加权融合
        fused = torch.mul(gate, out_x) + torch.mul(gate, out_y) + torch.mul(gate, out_z)

        # 融合特征通过全连接层生成最终输出
        output = self.fc_out(fused)

        return  output,inp ,FeatureInfo

class DeepTTC:
    def __init__(self, modeldir):

        model_drug = MolecularGCN(in_feats=75, dim_embedding=128,
                                           padding=True,
                                           hidden_feats=[128, 128, 128])
        # model_gene = GeneModel1()
        model_gene1 = MLP()
        model_prot = MLP2()
        model_cancer = MLP3()

        model_fusion = GatedFusionThreeModalities()

        # 构建网络模型
        self.model = Classifier(model_drug=model_drug, model_gene=model_gene1,model_prot=model_prot,model_cancer=model_cancer,model_fusion=model_fusion)

        self.device = torch.device('cuda:0')

        self.modeldir = modeldir
        self.record_file = os.path.join(self.modeldir, "valid_markdowntable.txt")  # 记录验证结果
        self.pkl_file = os.path.join(self.modeldir, "loss_curve_iter.pkl")  # 记录loss

    def test(self, datagenerator, model,epo):
        model.eval()
        for i, (v_drug, v_gene,v_prot,v_cancer, label , identifier) in enumerate(datagenerator):
            # score = model(v_drug, v_gene,v_prot,v_cancer)
            state = True
            train_state = True
            score , inp, FeatureInfo  = model(v_drug, v_gene, v_prot, v_cancer, label,epo,state,identifier,train_state)
            n = torch.squeeze(score, 1)

            onehot_labels_tr = one_hot_tensor(label, 2)
            loss_fct = FocalLoss()
            loss = loss_fct(score, onehot_labels_tr.to(device))
            for view in range(len(inp)):
                totalLoss = loss + torch.mean(FeatureInfo[view])

            loss = totalLoss


            # label1 = torch.unsqueeze(label,1).float().to(device)
            # loss_fct = nn.BCELoss()
            # loss = loss_fct(score,label1)

            # loss_fct = torch.nn.MSELoss()
            # loss = loss_fct(n, Variable(torch.from_numpy(np.array(label)).float()).to(device))
            test_label = Variable(torch.from_numpy(np.array(label))).float().to(device).to('cpu').numpy()
            test_predict = n.detach().cpu().numpy()
            test_predict = test_predict[:, 1]
            # test_label = torch.from_numpy(np.array(label)).float().numpy()
            # roc_auc_score(labels_trte[trte_idx["te"]], te_prob[:, 1]))
            test_predict = np.around(test_predict, decimals=2)
            fpr, tpr, thresholds1 = metrics.roc_curve(test_label, test_predict, pos_label=1)
            # print('AUC的threshold',thresholds)
            youden_index=tpr-fpr
            optimal_threshold1=thresholds1[np.argmax(youden_index)]
            print('AUC的optimal_threshold',optimal_threshold1)
            AUC = metrics.auc(fpr, tpr)
            precision, recall, thresholds2 = metrics.precision_recall_curve(test_label, test_predict)

            f1_scores = 2 * (precision * recall) / (precision + recall)
            optimal_threshold2 = thresholds2[np.argmax(f1_scores)]

            print("最佳 F1-score 阈值:", optimal_threshold2)

            predictions = (test_predict >= optimal_threshold2).astype(int)
            accuracy = accuracy_score(test_label, predictions)

            print(f"模型在阈值 0.49 下的准确率: {accuracy:.2%}")

            # 计算所有阈值下的precision、recall
            precision, recall, thresholds = metrics.precision_recall_curve(test_label, test_predict)

            # 因为precision_recall_curve的阈值数量比precision/recall少1，要补齐长度
            thresholds = np.append(thresholds, 1.0)

            accs = []
            for t in thresholds:
                preds = (test_predict >= t).astype(int)
                acc = (preds == test_label).mean()
                accs.append(acc)
            # 找到acc最接近0.8的那个阈值
            accs = np.array(accs)
            target_acc = 0.87
            idx = np.abs(accs - target_acc).argmin()
            threshold_acc_08 = thresholds[idx]

            print("acc最接近0.87时的阈值为：", threshold_acc_08)
            print("对应的acc为：", accs[idx])


            PRC = metrics.auc(recall, precision)
            score = score.detach().cpu().numpy()
            y_pred = (score[:, 1] >= 0.5).astype(int)
            label_numpy = label.detach().cpu().numpy()
            ACC = metrics.accuracy_score(label_numpy,y_pred)

            test_predict_class = (test_predict >= 0.5).astype(int)
            test_label = test_label.astype(int)
            misclassified_negative_samples = (test_label == 0) & (test_predict_class == 1)
            misclassified_positive_samples = (test_label == 1) & (test_predict_class == 0)
            # identifier_np = identifier.numpy()
            # ---------- Mann-Whitney U 检验 (p-value) ── NEW ----------
            neg_scores = test_predict[test_label == 0]
            pos_scores = test_predict[test_label == 1]

            statistic, pvalue = stats.mannwhitneyu(
                neg_scores, pos_scores,
                alternative='two-sided',
                use_continuity=True
            )

            print(f"p-value (Mann-Whitney U): {pvalue:.4e}")
            precision = precision[1:]
            recall = recall[1:]
            # thresholds现在和precision长度一致

            # 找到precision接近0.8的那个阈值
            target_precision = 0.70
            idx = np.abs(precision - target_precision).argmin()
            threshold_at_precision_08 = thresholds[idx]

            print("Precision=0.70时最接近的阈值为：", threshold_at_precision_08)
            print("对应的precision：", precision[idx])
            print("对应的recall：", recall[idx])
             # 提取 misclassified_negative_samples 为 True 的 identifier 值
            # misclassified_identifiers = identifier_np[misclassified_negative_samples]
            # misclassified_identifiers_pos = identifier_np[misclassified_positive_samples]
            # df = pd.DataFrame(misclassified_identifiers, columns=["identifier"])

            # 保存到 CSV 文件
            # df.to_csv("E:\研1\zh\Gene-CDR_q\Gene-CDR_q-main_oncokb\data\cgi_data\去重版new_300\delete380\misclassified_identifiers.csv", index=False)

            # print("误分类负样本的 identifier 值已保存到 misclassified_identifiers.csv")
        model.train()


        return loss,AUC,PRC,ACC,score,test_predict,thresholds1,test_label,pvalue
               # mean_squared_error(y_label, y_pred), \
               # np.sqrt(mean_squared_error(y_label, y_pred)), \
               # pearsonr(y_label, y_pred)[0], \
               # pearsonr(y_label, y_pred)[1], \
               # spearmanr(y_label, y_pred)[0], \
               # spearmanr(y_label, y_pred)[1], \
               # concordance_index(y_label, y_pred), \


    def train(self, train_drug, train_rna,train_prot,train_cancer, val_drug, val_rna,val_prot,val_cancer):
        lr = 1e-4
        decay = 0
        BATCH_SIZE = 32
        test_batch_size=3000
        train_epoch = 15
        final_max_roc_auc=0
        final_max_prc=0
        self.model = self.model.to(device)  # 使用GPU进行训练
        # self.model = torch.nn.DataParallel(self.model, device_ids=[0, 5])
        opt = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=decay)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=opt,
        #                                                        T_max=train_epoch)
        loss_history = []

        params = {'batch_size': BATCH_SIZE,
                  'shuffle': True,
                  'num_workers': 0,
                  'drop_last': False}
        params1 = {'batch_size': test_batch_size,
                  'shuffle': True,
                  'num_workers': 0,
                  'drop_last': False}
        # 加载数据
        # 加载数据

        def custom_collate_fn(batch):
            """
            自定义 collate 函数，用于批处理 DGLGraph 和其他数据类型。
            """
            # 解包 batch，batch 是一个列表，包含多个样本
            graphs, v_genes, v_prots, v_cancers, labels, identifiers = zip(*batch)

            # 批处理 DGLGraph
            batched_graph = dgl_batch(graphs)

            # 将其他数据转换为张量
            v_genes = torch.tensor(np.array(v_genes))  # 转为张量
            v_prots = torch.tensor(np.array(v_prots))  # 转为张量
            v_cancers = torch.tensor(np.array(v_cancers))  # 转为张量
            labels = torch.tensor(np.array(labels))  # 转为张量
            identifiers = list(identifiers)  # 直接保留为列表

            return batched_graph, v_genes, v_prots, v_cancers, labels, identifiers

        training_generator = data.DataLoader(data_process_loader(
            train_drug.index.values, train_drug.Label.values, train_drug, train_rna,train_prot,train_cancer), **params,collate_fn=custom_collate_fn)
        validation_generator = data.DataLoader(data_process_loader(
            val_drug.index.values, val_drug.Label.values, val_drug, val_rna,val_prot,val_cancer), **params1,collate_fn=custom_collate_fn)


        print('--- Go for Training ---')

        for epo in range(train_epoch):
            for i, (v_d, v_p, v_pt,v_c, label , identifier) in enumerate(training_generator):
                # print(v_d,v_p)
                # v_d = v_d.float().to(self.device)
                label = label
                state = False
                train_state = True
                score , inp, FeatureInfo = self.model(v_d, v_p, v_pt,v_c,label,epo,state,identifier,train_state)  # 64,1:batch_size为64，每个基因和药物对计算出一个分数（即IC50值）
                # label = Variable(torch.from_numpy(np.array(label))).float().to(device)
                # loss_fct = torch.nn.MSELoss()
                # n = torch.squeeze(score, 1).float()  # 压缩维度，把其中大小为1的维删除
                # loss = loss_fct(n, label)

                # label1 = torch.unsqueeze(label,1).float().to(device)
                # loss_fct = nn.BCELoss()
                # loss = loss_fct(score, label1)


                # MMLoss = torch.mean(criterion(score, label))
                # onehot_labels_tr = one_hot_tensor(label, 2)
                # label = label.to(device)
                # loss_fct = FocalLoss()
                onehot_labels_tr = one_hot_tensor(label, 2)
                label = label.to(device)
                loss_fct = FocalLoss()
                loss = loss_fct(score, onehot_labels_tr.to(device))
                for view in range(len(inp)):
                    totalLoss = loss + torch.mean(FeatureInfo[view])

                loss = totalLoss
                opt.zero_grad()
                loss.backward()
                opt.step()

                # label1 = torch.unsqueeze(label,1).float().to(device)
                # loss_fct = FocalLoss()
                # loss = loss_fct(score, label1)

            # scheduler.step()
                # if (i % 10000==0):
                #     # print("此时i为{}".format(i))# 每个epoch迭代一千次后进行数输出loss
                #     t_now = time.time()
                #     print('Training at Epoch ' + str(epo + 1) +
                #           ' iteration ' + str(i) + \
                #           ' with loss ' + str(loss.cpu().detach().numpy())[:7] + \
                #           ". Total time " + str(int(t_now - t_start) / 3600)[:7] + " hours")

            with torch.set_grad_enabled(False):  # 锁定梯度不再进行求导，进行模型的测试。
                loss_val,AUC,PRC,ACC,score,test_predict,thresholds,test_label,pvalue= self.test(validation_generator,self.model,epo)
                # if (final_max_roc_auc < AUC):
                #     final_max_roc_auc=AUC
                final_max_roc_auc = AUC
                    # torch.save(self.model.state_dict(), self.modeldir + '/42_2+2_model.pt')
                # if (final_max_prc<PRC):
                #     final_max_prc=PRC
                final_max_prc = PRC

                print("test_loss:"+str(loss_val.item())[:7])
                print("AUC:"+str(AUC))
                print("ACC:" + str(ACC))
                print("epoch"+str(epo))
        # return final_max_roc_auc,final_max_prc,pvalue

        return final_max_roc_auc, final_max_prc, pvalue, test_label, test_predict
    def test_no_loss(self, datagenerator, model,epo):
        model.eval()
        # "E:\研1\zh\Gene-CDR_q\Gene-CDR_q-main_oncokb\data\cell/feature\cell-oncokb\cell_smiles.csv"
        # output_file="data\data_new/0707/predict.csv"
        # first_batch = not os.path.exists(output_file)
        for i, (v_drug, v_gene,v_prot,v_cancer, label , identifier) in enumerate(datagenerator):
            # score = model(v_drug, v_gene,v_prot,v_cancer)
            state = True
            train_state = True
            score , inp, FeatureInfo  = model(v_drug, v_gene, v_prot, v_cancer, label,epo,state,identifier,train_state)
            n = torch.squeeze(score, 1)




            # test_label = Variable(torch.from_numpy(np.array(label))).float().to(device).to('cpu').numpy()
            test_predict = n.detach().cpu().numpy()
            test_predict = test_predict[:, 1]
            # test_label = torch.from_numpy(np.array(label)).float().numpy()
            # roc_auc_score(labels_trte[trte_idx["te"]], te_prob[:, 1]))
            test_predict = np.around(test_predict, decimals=4)

            score = score.detach().cpu().numpy()
            # 将标签转换为 numpy 数组
            test_label = label.detach().cpu().numpy()
            # 计算 AUROC
            try:
                auroc = roc_auc_score(test_label, test_predict)
                print(f"AUROC: {auroc:.4f}")
            except ValueError:
                print("AUROC cannot be calculated. Check label or prediction shape.")

            # 计算 AUPRC
            try:
                auprc = average_precision_score(test_label, test_predict)
                print(f"AUPRC: {auprc:.4f}")
            except ValueError:
                print("AUPRC cannot be calculated. Check label or prediction shape.")

            # 1) 按标签拆分预测分数
            neg_scores = test_predict[test_label == 0]
            pos_scores = test_predict[test_label == 1]

            statistic, pvalue = stats.mannwhitneyu(
                neg_scores, pos_scores,
                alternative='two-sided',
                use_continuity=True
            )

            print(f"p-value (Mann-Whitney U): {pvalue:.4e}")
            # 构建 DataFrame
            df = pd.DataFrame({"Identifier": identifier, "Score": test_predict})

            # 追加写入 CSV
            # df.to_csv(output_file, mode='a', index=False, header=first_batch)

            # 只在第一批次写入表头，之后的批次就不写了
            first_batch = False

        # 创建 DataFrame 并保存
        # df = pd.DataFrame({"Identifier": all_identifiers, "Score": all_scores})
        df.to_csv("E:\研1\zh\Gene-CDR_q\Gene-CDR_q-main_oncokb\data\civic_data\与oncokb去重版\drop_duplicates\prediction_results.csv", index=False)

        print("✅ 预测结果已保存至 prediction_results.csv")

        return score,test_predict,identifier


    def predict(self, val_drug, val_rna, val_prot, val_cancer):
        def custom_collate_fn(batch):
            """
            自定义 collate 函数，用于批处理 DGLGraph 和其他数据类型。
            """
            # 解包 batch，batch 是一个列表，包含多个样本
            graphs, v_genes, v_prots, v_cancers, labels, identifiers = zip(*batch)

            # 批处理 DGLGraph
            batched_graph = dgl_batch(graphs)

            # 将其他数据转换为张量
            v_genes = torch.tensor(np.array(v_genes))  # 转为张量
            v_prots = torch.tensor(np.array(v_prots))  # 转为张量
            v_cancers = torch.tensor(np.array(v_cancers))  # 转为张量
            labels = torch.tensor(np.array(labels))  # 转为张量
            identifiers = list(identifiers)  # 直接保留为列表

            return batched_graph, v_genes, v_prots, v_cancers, labels, identifiers
        print('predicting...')
        epo=0
        test_batch_size = 3000
        self.model.to(device)

        params1 = {'batch_size': test_batch_size,
                   'shuffle': False,
                   'num_workers': 0,
                   'drop_last': False}
        validation_generator = data.DataLoader(data_process_loader(
            val_drug.index.values, val_drug.Label.values, val_drug, val_rna,val_prot, val_cancer), **params1,collate_fn=custom_collate_fn)

        score,test_predict,identifier = self.test_no_loss(validation_generator, self.model,epo)
        # y_pred = model.predict(X_test)  # 获取预测的类别标签
        # y_pred_prob = model.predict_proba(X_test)[:, 1]  # 获取正类的预测概率
        threshold = 0.49

        # 将概率转换为类别预测
        test_predict_class = (test_predict >= threshold).astype(int)
        # test_label=test_label.astype(int)
        # misclassified_negative_samples = (test_label == 0) & (test_predict_class == 1)
        return score,test_predict,identifier
    # def save_model(self):
    #     torch.save(self.model.state_dict(), self.modeldir + '/model.pt')

    def save_model(self,modelfile):
        torch.save(self.model.state_dict(), modelfile )

    def load_pretrained(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

        if self.device == 'cuda':
            state_dict = torch.load(path)
        else:
            state_dict = torch.load(path, map_location=torch.device('cpu'))

        if next(iter(state_dict))[:7] == 'module.':
            # the pretrained model is from data-parallel module
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            state_dict = new_state_dict

        self.model.load_state_dict(state_dict)

class SHAPModelWrapper(nn.Module):
    def __init__(self, model):
        super(SHAPModelWrapper, self).__init__()
        self.model = model

    def forward(self, v_D, v_P,v_Pt,v_C):

        # v_D = v_D.to(device)
        # v_D = self.model_drug(v_D)#.to(torch.float64)  # v_D->64,128
        v_P=v_P.to(device)
        v_P1 = self.model.model_gene(v_P)
        v_Pt = v_Pt.to(device)
        v_Pt_1 = self.model.model_prot(v_Pt)
        v_C = v_C.to(device)
        v_C1 = self.model.model_cancer(v_C)

        v_fusion_out,inp,FeatureInfo = self.model.model_fusion(v_P1,v_Pt_1,v_C1)

        v_d_mean = torch.mean(v_D, dim=1)
        v_f = torch.cat((v_d_mean, v_fusion_out), 1)


        for i, l in enumerate(self.model.predictor):
            if i == (len(self.model.predictor) - 1):
                v_f = l(v_f)
                v_f = self.model.act(v_f)

            else:
                v_f = v_f.float()

                v_f = F.relu(self.model.dropout(l(v_f)))


        return v_f

class GradModelWrapper(nn.Module):
    def __init__(self, model):
        super(GradModelWrapper, self).__init__()
        self.model = model

    def forward(self, v_D, v_P,v_Pt,v_C):

        # v_D = v_D.to(device)
        # v_D = self.model_drug(v_D)#.to(torch.float64)  # v_D->64,128
        v_D = v_D.to(device)
        v_D = self.model.model_drug(v_D)
        v_P=v_P.to(device)
        v_P1 = self.model.model_gene(v_P)
        v_Pt = v_Pt.to(device)
        v_Pt_1 = self.model.model_prot(v_Pt)
        v_C = v_C.to(device)
        v_C1 = self.model.model_cancer(v_C)

        v_fusion_out,inp,FeatureInfo = self.model.model_fusion(v_P1,v_Pt_1,v_C1)
        # 将 v_fusion_out 转换为 (1, 192)
        v_fusion_out = v_fusion_out.unsqueeze(0)

        v_d_mean = torch.mean(v_D, dim=1)
        v_f = torch.cat((v_d_mean, v_fusion_out), 1)


        for i, l in enumerate(self.model.predictor):
            if i == (len(self.model.predictor) - 1):
                v_f = l(v_f)
                v_f = self.model.act(v_f)

            else:
                v_f = v_f.float()

                v_f = F.relu(self.model.dropout(l(v_f)))


        return v_f

def c_v(drug_data,gene_data,prot_data,cancer_data):
    sum=0
    sum1=0
    pvalue_sum=0
    i=1
    all_val_labels = []
    all_val_preds = []
    for fold, (train_index, test_index) in enumerate(skf.split(drug_data, drug_data.loc[:, 'Label']), 1):
        # if fold == 3:  # 选择第三折
        print(f"Running fold {fold}...")
    # print("第",i,"折进行训练")
        train_drug = drug_data.iloc[train_index].reset_index()
        test_drug = drug_data.iloc[test_index].reset_index()
        train_gene = gene_data.iloc[train_index]
        test_gene = gene_data.iloc[test_index]
        train_prot = prot_data.iloc[train_index]
        test_prot = prot_data.iloc[test_index]
        train_cancer = cancer_data.iloc[train_index]
        test_cancer = cancer_data.iloc[test_index]

        modeldir = 'Model'
        # modelfile = f'{modeldir}/gate_sparse_fusion/oncokb_drugban_dnabert2_3_gcn_gate_protein_sparse_epoch15_fold_{i}.pt'


        if not os.path.exists(modeldir):
            os.mkdir(modeldir)

        net = DeepTTC(modeldir=modeldir)
        final_max_roc_auc,final_max_prc,pvalue,labels_this_fold,preds_this_fold=net.train(train_drug=train_drug, train_rna=train_gene,train_prot=train_prot,train_cancer=train_cancer,
                  val_drug=test_drug, val_rna=test_gene,val_prot=test_prot,val_cancer=test_cancer)
        # net.load_pretrained("Model/gate_sparse_fusion/oncokb_drugban_dnabert2_3_gcn_gate_protein_sparse_epoch15_fold_3.pt")
        # net.load_pretrained('Model/gate_sparse_fusion/oncokb_drugban_alphafold3_3_gcn_gate_protein_sparse_epoch15_fold_3_normalized.pt')
        # net.to(device)

        # 记录单折
        all_val_labels.append(labels_this_fold)
        all_val_preds.append(preds_this_fold)
        # ...统计AUC/PRC/pvalue
        sum=sum+final_max_roc_auc
        sum1=sum1+final_max_prc
        pvalue_sum=pvalue_sum+pvalue


        # net.save_model(modelfile)
        # print("Model Saved :{}".format(modelfile))



        print("______________________")
        print("第",i,"折的final_max_roc_auc"+str(final_max_roc_auc))
        print("第",i,"折的final_max_prc"+str(final_max_prc))

        i = i + 1
        print("-----------------------")
    print("交叉验证的平均auc为:" + str(sum / 5))
    print("交叉验证的平均prc为:" + str(sum1 / 5))
    print("交叉验证的平均pvalue为:" + str(pvalue_sum / 5))
    # 循环外，拼接并计算阈值
    all_val_labels = np.concatenate(all_val_labels)
    all_val_preds = np.concatenate(all_val_preds)
    precision, recall, thresholds = precision_recall_curve(all_val_labels, all_val_preds)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    best_idx = np.argmax(f1)
    best_threshold = thresholds[best_idx]
    print("五折拼接后F1-score最大时的全局阈值：", best_threshold)
    # ...均值输出



def test_on_other_data(drug_data,gene_data,prot_data,cancer_data):


    modeldir = 'Model'
    net = DeepTTC(modeldir=modeldir)
    net.load_pretrained("Model/gate_sparse_fusion/oncokb_drugban_dnabert2_3_gcn_gate_protein_sparse_epoch15_fold_3.pt")
    score,test_predict,identifier= net.predict(drug_data, gene_data, prot_data, cancer_data)
    # print("验证的auc为:" + str(auc))
    # print("验证的prc为:" + str(prc))






def predict_c_v(drug_data,gene_data):
    sum=0
    sum1=0
    i=1
    modeldir = 'Model'
    modelfile = modeldir + '/model.pt'
    if not os.path.exists(modeldir):
        os.mkdir(modeldir)

    net = DeepTTC(modeldir=modeldir)
    print("-----------------------")
    net.load_pretrained("D:\machine-learning\DeepTTC-main - 副本1\Model/42_2+2_model.pt")
    for train_index,test_index in skf.split(drug_data,drug_data.loc[:,'Label']):
        # print('Train Index:',train_index,'Test Index:',test_index)
        print("第",i,"折使用最佳模型进行测试")
        train_drug = drug_data.iloc[train_index].reset_index()
        test_drug = drug_data.iloc[test_index].reset_index()
        train_gene = gene_data.iloc[train_index]
        test_gene = gene_data.iloc[test_index]
        auc = net.predict(test_drug,test_gene)
        print("测试验证的auc为:"+str(auc))
        i = i + 1
        sum=sum+auc
    print("测试交叉验证的平均auc为:"+str(sum/5))
    print("测试交叉验证的平均prc为:"+str(sum1/5))

if __name__ == '__main__':

    # # step1 数据切分
    from Step2_DataEncoding import DataEncoding

    vocab_dir = 'E:\研1\zh\Gene-CDR_q\Gene-CDR_q-main_oncokb'
    obj = DataEncoding(vocab_dir=vocab_dir)

    dataset_paths = [
        {
            # 1. CGI 去重版 (delete380/drop_duplicates)
            "data": "data/cgi_data/去重版new_300/delete380/drop_duplicates/cgi_get_drugid.csv",
            "gene": "data/cgi_data/去重版new_300/delete380/drop_duplicates/cgi_dnabert2.csv",
            "prot": "data/cgi_data/去重版new_300/delete380/drop_duplicates/cgi_protein_dim64_pos_neg_new_drop_duplicates_v2.csv",
            "cancer": "data/cgi_data/去重版new_300/delete380/drop_duplicates/cgi_cancerfeature_pos_neg_new_drop_duplicates_v2.csv"
        },
        {
            # 2. CIViC 与oncokb去重版 (drop_duplicates)
            "data": "data/civic_data/与oncokb去重版/drop_duplicates/civic_get_drugid.csv",
            "gene": "data/civic_data/与oncokb去重版/drop_duplicates/civic_dnabert2.csv",
            "prot": "data/civic_data/与oncokb去重版/drop_duplicates/civic_protein_dim64_pos_neg.csv",
            "cancer": "data/civic_data/与oncokb去重版/drop_duplicates/civic_cancerfeature_pos_neg.csv"
        },
        {
            # 3. COSMIC 与oncokb去重版
            "data": "data/cosmic_data/与oncokb去重版/cosmic_get_drugid_v2.csv",
            "gene": "data/cosmic_data/与oncokb去重版/cosmic_dnabert2.csv",
            "prot": "data/cosmic_data/与oncokb去重版/cosmic_protein_dim64_pos_neg_new_v2.csv",
            "cancer": "data/cosmic_data/与oncokb去重版/cosmic_cancerfeature_pos_neg_new_v2.csv"
        },
        {
            # 4. MetaKB
            "data": "data/metaKB/metaKB_get_drugid.csv",
            "gene": "data/metaKB/metaKB_dnabert2.csv",
            "prot": "data/metaKB/metaKB_protein_dim64.csv",
            "cancer": "data/metaKB/metaKB_pancan.csv"
        }
    ]

    # 用于存储读取后的DataFrame
    train_dfs = []
    test_dfs = []
    gene_dfs = []
    prot_dfs = []
    cancer_dfs = []

    print("Start loading and merging datasets...")

    for i, paths in enumerate(dataset_paths):
        print(f"Loading dataset {i + 1}...")

        # --- 读取 Train/Test 数据 ---
        # 假设 train 和 test 读取的是同一个文件 (根据原代码逻辑)
        curr_train = pd.read_csv(paths["data"])
        curr_test = pd.read_csv(paths["data"])  # 原代码中test和train读取路径一致

        # 删除 'source' 列 (如果存在)
        if 'source' in curr_train.columns:
            curr_train = curr_train.drop(columns=['source'])
        if 'source' in curr_test.columns:
            curr_test = curr_test.drop(columns=['source'])

        train_dfs.append(curr_train)
        test_dfs.append(curr_test)

        # --- 读取 特征 数据 ---
        gene_dfs.append(pd.read_csv(paths["gene"]))
        prot_dfs.append(pd.read_csv(paths["prot"]))
        cancer_dfs.append(pd.read_csv(paths["cancer"]))

    # ==========================================
    # 2. 合并数据 (纵向合并，自动对齐列名)
    # ==========================================
    # ignore_index=True 保证索引重置，合并时列名会自动对齐
    traindata = pd.concat(train_dfs, axis=0, ignore_index=True)
    testdata = pd.concat(test_dfs, axis=0, ignore_index=True)

    Gene_df = pd.concat(gene_dfs, axis=0, ignore_index=True)
    Prot_df = pd.concat(prot_dfs, axis=0, ignore_index=True)
    Cancer_df = pd.concat(cancer_dfs, axis=0, ignore_index=True)

    print(f"Total Traindata shape: {traindata.shape}")
    print(f"Total Gene_df shape: {Gene_df.shape}")

    # ==========================================
    # 3. 数据编码 (Encode)
    # ==========================================
    traindata, testdata = obj.encode(
        traindata=traindata,
        testdata=testdata
    )

    # ==========================================
    # 4. 特征归一化 (Normalization)
    # ==========================================
    eps = 1e-8

    # Gene Normalization
    min_gene = Gene_df.min()
    max_gene = Gene_df.max()
    Gene_df_normalized = (Gene_df - min_gene) / (max_gene - min_gene + eps)

    # Protein Normalization
    min_prot = Prot_df.min()
    max_prot = Prot_df.max()
    Prot_df_normalized = (Prot_df - min_prot) / (max_prot - min_prot + eps)

    # Cancer Feature Normalization
    min_cancer = Cancer_df.min()
    max_cancer = Cancer_df.max()
    # Cancer_df_normalized = (Cancer_df - min_cancer) / (max_cancer - min_cancer + eps)
    c_v(drug_data=traindata, gene_data=Gene_df, prot_data=Prot_df, cancer_data=Cancer_df)
    # ==========================================
    # 5. 模型测试
    # ==========================================
    # 传入合并后的大表
    # test_on_other_data(traindata, Gene_df, Prot_df, Cancer_df)
