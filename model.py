from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from typing import Any, Union, List
from simple_tokenizer import SimpleTokenizer as _Tokenizer
from torch import nn, optim

from torch_geometric.nn import GCNConv


_tokenizer = _Tokenizer()


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))

        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):

        return self.resblocks(x)


class GraphConvModel(torch.nn.Module):
    def __init__(self, in_ft, hid_ft, out_ft, num_layers):
        super(GraphConvModel, self).__init__()
        self.num_layers = num_layers
        self.conv_layers = nn.ModuleList()
        
        # Input layer
        self.conv_layers.append(GCNConv(hid_ft, hid_ft))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.conv_layers.append(GCNConv(hid_ft, hid_ft))
        
        # Output layer
        self.conv_layers.append(GCNConv(hid_ft, hid_ft))
        
        self.res = nn.Linear(in_ft, out_ft)

        self.linear = nn.Linear(hid_ft, out_ft)

        self.proj = nn.Linear(in_ft, hid_ft) 


        self.act = nn.PReLU()

    def drop_feature(self, x, drop_prob):
        drop_mask = torch.empty((x.size(1),),
                            dtype=torch.float32,
                            device=x.device).uniform_(0, 1) < drop_prob
        
        x_aug = x.clone()
        x_aug[:, drop_mask] = 0
        return x_aug

    def forward(self, x, edge_index):
        ori_x = x
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.proj(x)
        xs = []
        xs.append(x)

        for i in range(self.num_layers):
            x_res = x
            x = self.conv_layers[i](x, edge_index)
            x = F.leaky_relu(x+x_res)
            x = F.dropout(x, p=0.2, training=self.training) 

        x = self.act(self.linear(x)) + self.act(self.res(ori_x)) 

        return x
    

class TAGAD(nn.Module):
    def __init__(self,
                 args
                 ):
        super().__init__()

        self.context_length = args.context_length
        self.args = args
        self.edge_coef = args.edge_coef

        self.gnn = GraphConvModel(args.gnn_input, args.gnn_hid, args.gnn_output, args.gnn_layers)

        self.transformer = Transformer(
            width=args.transformer_width,
            layers=args.transformer_layers,
            heads=args.transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = args.vocab_size
        self.token_embedding = nn.Embedding(args.vocab_size,
                                            args.transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, args.transformer_width))

        self.ln_final = LayerNorm(args.transformer_width)

        self.text_projection = nn.Parameter(torch.empty(args.transformer_width, args.embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / args.scale)) 

        self.dtype = torch.float32
        
        self.optim = optim.Adam([{'params': self.token_embedding.weight},
                                {'params': self.positional_embedding},
                                {'params': self.transformer.parameters()},
                                {'params': self.text_projection},
                                {'params': self.logit_scale}, 
                                {'params': self.gnn.parameters()}
                                ], lr=args.lr)


        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)


    def build_attention_mask(self):
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))

        mask.triu_(1)

        return mask

    def encode_graph(self, idx_train, x, adj):
        embs = self.gnn(x, adj)

        train_embs = embs[idx_train]
        return train_embs, embs

    def encode_text(self, text):

        x = self.token_embedding(text).type(self.dtype)  

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)

        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]

        x = x @ self.text_projection

        return x
    
    def drop_feature(self, x, drop_prob):
        drop_mask = torch.empty((x.size(1),),
                            dtype=torch.float32,
                            device=x.device).uniform_(0, 1) < drop_prob
        
        x_aug = x.clone()
        x_aug[:, drop_mask] = 0
        return x_aug


    def forward(self, x, adj, s_n, t_n, s_n_text, t_n_text, epoch, device, training=True):
        s_graph_features, embs = self.encode_graph(s_n, x, adj)

        t_n_arr = t_n.numpy().reshape(-1)

        t_graph_features = embs[t_n_arr]

        s_text_features = self.encode_text(s_n_text)
        t_text_features = self.encode_text(t_n_text)


        t_text_features = t_text_features.reshape(s_graph_features.shape[0], self.args.neigh_num, self.args.gnn_output)

        t_text_features = torch.mean(t_text_features, dim=1, keepdim=False)
        
        t_graph_features = t_graph_features.reshape(s_graph_features.shape[0], self.args.neigh_num, self.args.gnn_output)
        t_graph_features = torch.mean(t_graph_features, dim=1, keepdim=False)


        # normalized features
        s_graph_features = s_graph_features / s_graph_features.norm(dim=-1, keepdim=True)
        t_graph_features = t_graph_features / t_graph_features.norm(dim=-1, keepdim=True)

        s_text_features = s_text_features / s_text_features.norm(dim=-1, keepdim=True)
        t_text_features = t_text_features / t_text_features.norm(dim=-1, keepdim=True)
      

        num_classes = s_graph_features.shape[0]
        labels = torch.arange(num_classes).to(device)


        logit_scale = self.logit_scale.exp()  


        logits1 = logit_scale * s_graph_features @ s_text_features.t()
        loss_i = F.cross_entropy(logits1, labels)
        loss_t = F.cross_entropy(logits1.T, labels)
        sg_st_loss = (loss_i + loss_t) / 2
        
        
        logits2 = logit_scale * s_graph_features @ t_text_features.t()
        loss_i = F.cross_entropy(logits2, labels)
        loss_t = F.cross_entropy(logits2.T, labels)
        sg_tt_loss = (loss_i + loss_t) / 2
 

        logits3 = logit_scale * t_graph_features @ s_text_features.t()
        loss_i = F.cross_entropy(logits3, labels)
        loss_t = F.cross_entropy(logits3.T, labels)
        tg_st_loss = (loss_i + loss_t) / 2
        

        logits4 = logit_scale * t_graph_features @ t_text_features.t()
        loss_i = F.cross_entropy(logits4, labels)
        loss_t = F.cross_entropy(logits4.T, labels)
        tg_tt_loss = (loss_i + loss_t) / 2
       

        logits5 = logit_scale * s_graph_features @ t_graph_features.t()
        loss_i = F.cross_entropy(logits5, labels)
        loss_t = F.cross_entropy(logits5.T, labels)
        sg_tg_loss = (loss_i + loss_t) / 2
     

        logits6 = logit_scale * s_text_features @ t_text_features.t()
        loss_i = F.cross_entropy(logits6, labels)
        loss_t = F.cross_entropy(logits6.T, labels)
        st_tt_loss = (loss_i + loss_t) / 2


        all_loss = sg_st_loss + sg_tt_loss + tg_st_loss + tg_tt_loss + (sg_tg_loss + st_tt_loss) * self.args.gamma


        if training == True:
            self.optim.zero_grad()
            torch.cuda.empty_cache()
            all_loss.backward()
            self.optim.step()
            if self.args.optim == 5 or self.args.optim == 6:
                self.scheduler.step()
            
        return round((all_loss.detach().clone()).cpu().item(), 4)
    
    def embedding(self, x, adj, s_n, t_n, s_n_text, t_n_text, device, training=True):

        s_graph_features, embs = self.encode_graph(s_n, x, adj)

        s_text_features = self.encode_text(s_n_text)

        return s_graph_features, s_text_features

    def inference(self, x, graph_feats, text_feats, adj, s_n, t_n, s_n_text, t_n_text, device, j, i_batch, training=True):

        s_n_arr, t_n_arr = s_n.numpy(), t_n.numpy().reshape(-1)

        s_graph_features = graph_feats[s_n_arr]

        t_graph_features = graph_feats[t_n_arr]


        s_text_features = text_feats[s_n_arr]
        t_text_features = text_feats[t_n_arr]


        t_text_features = t_text_features.reshape(s_graph_features.shape[0], self.args.neigh_num, self.args.gnn_output)
        t_text_features = torch.mean(t_text_features, dim=1, keepdim=False)

        
        t_graph_features = t_graph_features.reshape(s_graph_features.shape[0], self.args.neigh_num, self.args.gnn_output)
        t_graph_features = torch.mean(t_graph_features, dim=1, keepdim=False)


        # normalized features
        s_graph_features = s_graph_features / s_graph_features.norm(dim=-1, keepdim=True)
        t_graph_features = t_graph_features / t_graph_features.norm(dim=-1, keepdim=True)

        s_text_features = s_text_features / s_text_features.norm(dim=-1, keepdim=True)
        t_text_features = t_text_features / t_text_features.norm(dim=-1, keepdim=True)


        labels = torch.arange(s_graph_features.shape[0]).to(device)
        logit_scale = self.logit_scale.exp()  # the temporature hyperparameter


        logits = logit_scale * s_graph_features @ s_text_features.t()

        pos_sim = torch.diag(logits)

        neg_sim_matrix = logits.clone()
        neg_sim_matrix.fill_diagonal_(float(0))

        neg_sim = neg_sim_matrix.mean(dim=1)
        neg_sim1 = neg_sim_matrix.T.mean(dim=1)

        res1 = neg_sim + neg_sim1 - pos_sim * 4 + (F.cross_entropy(logits, labels, reduction='none') + F.cross_entropy(logits.T, labels, reduction='none')) / 4


        logits = logit_scale * s_graph_features @ t_text_features.t()
        pos_sim = torch.diag(logits)

        neg_sim_matrix = logits.clone()
        neg_sim_matrix.fill_diagonal_(float(0))

        neg_sim = neg_sim_matrix.mean(dim=1)
        neg_sim1 = neg_sim_matrix.T.mean(dim=1)

        res2 = neg_sim + neg_sim1 - pos_sim * 4 + (F.cross_entropy(logits, labels, reduction='none') + F.cross_entropy(logits.T, labels, reduction='none')) / 4


        logits = logit_scale * t_graph_features @ s_text_features.t()

        pos_sim = torch.diag(logits)
        neg_sim_matrix = logits.clone()
        neg_sim_matrix.fill_diagonal_(float(0))

        neg_sim = neg_sim_matrix.mean(dim=1)
        neg_sim1 = neg_sim_matrix.T.mean(dim=1)

        res3 = neg_sim + neg_sim1 - pos_sim * 4 + (F.cross_entropy(logits, labels, reduction='none') + F.cross_entropy(logits.T, labels, reduction='none')) / 4


        logits = logit_scale * t_graph_features @ t_text_features.t()

        pos_sim = torch.diag(logits)

        neg_sim_matrix = logits.clone()
        neg_sim_matrix.fill_diagonal_(float(0))

        neg_sim = neg_sim_matrix.mean(dim=1)
        neg_sim1 = neg_sim_matrix.T.mean(dim=1)
       
        res4 = neg_sim + neg_sim1 - pos_sim * 4 + (F.cross_entropy(logits, labels, reduction='none') + F.cross_entropy(logits.T, labels, reduction='none')) / 4


        logits = logit_scale * s_graph_features @ t_graph_features.t()

        pos_sim = torch.diag(logits)

        neg_sim_matrix = logits.clone()
        neg_sim_matrix.fill_diagonal_(float(0))

        neg_sim = neg_sim_matrix.mean(dim=1)
        neg_sim1 = neg_sim_matrix.T.mean(dim=1)
      
        res5 = neg_sim + neg_sim1 - pos_sim * 4 + (F.cross_entropy(logits, labels, reduction='none') + F.cross_entropy(logits.T, labels, reduction='none')) / 4


        logits = logit_scale * s_text_features @ t_text_features.t()

        pos_sim = torch.diag(logits)

        neg_sim_matrix = logits.clone()
        neg_sim_matrix.fill_diagonal_(float(0))

        neg_sim = neg_sim_matrix.mean(dim=1)
        neg_sim1 = neg_sim_matrix.T.mean(dim=1)

        res6 = neg_sim + neg_sim1 - pos_sim * 4 + (F.cross_entropy(logits, labels, reduction='none') + F.cross_entropy(logits.T, labels, reduction='none')) / 4


        ano_score = res1 + res2 + res3 + res4 + (res5 + res6) * self.args.gamma
        
        return ano_score


def tokenize(texts: Union[str, List[str]], context_length: int = 128, truncate: bool = True) -> torch.LongTensor:

    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]

    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")


        
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result
