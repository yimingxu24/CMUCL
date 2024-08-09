from torch.utils.data import DataLoader
from sklearn import preprocessing
import numpy as np
import pandas as pd
import argparse
import torch
from random import sample
import random
import time
from model import CMUCL, tokenize
from data import DataHelper
from sklearn import preprocessing
import dgl

import scipy.sparse as sp
import scipy.io as sio
import networkx as nx
from sklearn.metrics import roc_auc_score


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main(args):
    setup_seed(seed)

    model = CMUCL(args).to(device)

    Data = DataHelper(arr_edge_index, args)

    model.train()

    for j in range(args.epoch_num):
        loader = DataLoader(Data, batch_size=args.batch_size, shuffle=True, num_workers=10)

        for i_batch, sample_batched in enumerate(loader):

            s_n, t_n = sample_batched['s_n'], sample_batched['t_n']

            s_n_arr, t_n_arr = s_n.numpy(), t_n.numpy().reshape(-1)

            s_n_text, t_n_text = tit_list[s_n_arr].tolist(), tit_list[t_n_arr].tolist()

            s_n_text, t_n_text = tokenize(s_n_text, context_length=args.context_length).to(device), tokenize(t_n_text, context_length=args.context_length).to(device)

            loss = model.forward(node_f, edge_index, s_n, t_n, s_n_text, t_n_text, j, device)

            if j == 0 and i_batch % 100 == 0:
                print('{}th loss in the first epoch:{}'.format(i_batch, loss))
            

        print('{}th epoch loss:{}'.format(j, loss))

    print('Pre-training is finished')

    multi_round_ano_score = torch.zeros(num_nodes, args.rounds).to(device)
    graph_features = torch.zeros((num_nodes, args.gnn_output)).to(device)
    text_features = torch.zeros((num_nodes, args.embed_dim)).to(device)

    model.eval()
    with torch.no_grad():
        start = time.perf_counter()

        infer_loader = DataLoader(Data, batch_size=args.batch_size, shuffle=True, num_workers=10)

        for i_batch, sample_batched in enumerate(infer_loader):
                s_n, t_n = sample_batched['s_n'], sample_batched['t_n']
                s_n_arr, t_n_arr = s_n.numpy(), t_n.numpy().reshape(-1)
                s_n_text, t_n_text = tit_list[s_n_arr].tolist(), tit_list[t_n_arr].tolist()

                s_n_text, t_n_text = tokenize(s_n_text, context_length=args.context_length).to(device), tokenize(t_n_text, context_length=args.context_length).to(device)
                s_graph_feats, s_text_feats = model.embedding(node_f, edge_index, s_n, t_n, s_n_text, t_n_text, device)

                graph_features[s_n] = s_graph_feats
                text_features[s_n] = s_text_feats

        end = time.perf_counter()
        print("embedding time consuming {:.2f}".format(end - start))

        start = time.perf_counter()

        for j in range(args.rounds):
            infer_loader = DataLoader(Data, batch_size=args.batch_size, shuffle=True, num_workers=10)
            for i_batch, sample_batched in enumerate(infer_loader):

                s_n, t_n = sample_batched['s_n'], sample_batched['t_n']

                ano_score = model.inference(node_f, graph_features, text_features, edge_index, s_n, t_n, s_n_text, t_n_text, device, j, i_batch)

                multi_round_ano_score[s_n, j] = ano_score
        end = time.perf_counter()
        print("inference time consuming {:.2f}".format(end - start))

      

    ano_score = torch.mean(multi_round_ano_score, axis=1) + torch.std(multi_round_ano_score, axis=1)

    ano_score = ano_score.cpu().numpy()


    auc = roc_auc_score(ano_label, ano_score)

    print('AUC:{:.4f}'.format(auc))



def load_mat(dataset, train_rate=0.3, val_rate=0.1):
    """Load .mat dataset."""
    data = sio.loadmat("./data/processed/{}/{}.mat".format(dataset,dataset))
    text = pd.read_csv('./data/processed/{}/{}.csv'.format(args.dataset, args.dataset))


    label = data['Label'] if ('Label' in data) else data['gnd']
    attr = data['Attributes'] if ('Attributes' in data) else data['X']
    network = data['Network'] if ('Network' in data) else data['A']
    
    adj = sp.csr_matrix(network)
    feat = sp.lil_matrix(attr)

    ano_labels = np.squeeze(np.array(label))

    return adj, feat, ano_labels, text

def adj_to_dgl_graph(adj):
    """Convert adjacency matrix to dgl format."""
    nx_graph = nx.from_scipy_sparse_matrix(adj)
    dgl_graph = dgl.DGLGraph(nx_graph)
    return dgl_graph

def preprocess_features_ndarray(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(axis=1))  # sum of each row
    r_inv = np.power(rowsum, -1).flatten()  # inverse of row sums
    r_inv[np.isinf(r_inv)] = 0.  # replace inf with 0
    r_mat_inv = np.diag(r_inv)  # create a diagonal matrix with r_inv
    features = r_mat_inv.dot(features)  # row-normalize the feature matrix

    return features

def position_encoding(max_len, emb_size):
    # pe = np.zeros((max_len, emb_size))
    # position = np.arange(0, max_len)[:, np.newaxis]

    pe = np.zeros((max_len, emb_size), dtype=np.float32)
    position = np.arange(0, max_len, dtype=np.float32)[:, np.newaxis]

    div_term = np.exp(np.arange(0, emb_size, 2) * -(np.log(10000.0) / emb_size))

    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    
    return pe

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--aggregation_times', type=int, default=2, help='Aggregation times')
    parser.add_argument('--epoch_num', type=int, default=2, help='epoch number')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--edge_coef', type=float, default=10)
    parser.add_argument('--neigh_num', type=int, default=3)
    parser.add_argument('--scale', type=float, default=0.07)


    parser.add_argument('--gnn_layers', type=int, default=2)
    parser.add_argument('--gnn_input', type=int, default=768) 
    parser.add_argument('--gnn_hid', type=int, default=128)
    parser.add_argument('--gnn_output', type=int, default=128)

    parser.add_argument('--context_length', type=int, default=128)

    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--transformer_heads', type=int, default=8)
    parser.add_argument('--transformer_layers', type=int, default=12)
    parser.add_argument('--transformer_width', type=int, default=512)
    parser.add_argument('--vocab_size', type=int, default=49408)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--data_name', type=str, default='cora')

    parser.add_argument('--num_workers_per_gpu', type=int, default=0)
    parser.add_argument('--norm', type=int, default=0)
    parser.add_argument('--reg', type=int, default=0)
    parser.add_argument('--drop_prob', type=float, default=0.2)
    parser.add_argument('--optim', type=int, default=0)
    
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--rounds', type=int, default=256)
    parser.add_argument('--seed', type=int, default=1)

    
    args = parser.parse_args()
    print(torch.cuda.device_count())
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    print('device:', device)

    num_nodes = 0
    tit_list = []

    adj, features, ano_label, text = load_mat(args.dataset)
    dgl_graph = adj_to_dgl_graph(adj)
    num_nodes = dgl_graph.number_of_nodes()
   
    arr_edge_index = np.vstack((dgl_graph.edges()[0].numpy(), dgl_graph.edges()[1].numpy()))
    edge_index = torch.stack(dgl_graph.edges()).to(device)

    node_f = features.toarray()

    node_f = preprocessing.StandardScaler().fit_transform(node_f)

    node_feat = torch.from_numpy(node_f).float()
    node_f = torch.from_numpy(node_f).to(device)
    
    tit_list = text['text'].to_numpy()

    start = time.perf_counter()

    seed = args.seed
    main(args)

    end = time.perf_counter()
    print("time consuming {:.2f}".format(end - start))
