import os
os.environ['MPLCONFIGDIR'] = "/tmp"
import random
import numpy as np
import torch
import logging

from graphgym.cmd_args import parse_args
from graphgym.config import (cfg, assert_cfg, dump_cfg,
                             update_out_dir, get_parent_dir)
from graphgym.loader import create_dataset, create_loader
from graphgym.logger import setup_printing, create_logger
from graphgym.optimizer import create_optimizer, create_scheduler
from graphgym.model_builder import create_model
from graphgym.train import train
from graphgym.utils.agg_runs import agg_runs
from graphgym.utils.comp_budget import params_count
from graphgym.utils.device import auto_select_device
from graphgym.contrib.train import *
from graphgym.register import train_dict

from deepsnap.graph import Graph
from deepsnap.dataset import GraphDataset

from datetime import datetime

import warnings
warnings.filterwarnings("ignore")

def dataset_cfg_setup_fixed_split(name: str):
    """
    Setup required fields in cfg for the given dataset.
    """
    if name in ['bsi_svt_2008.tsv', 'bsi_zk_2008.tsv']:
        cfg.dataset.format = 'roland_bsi_general'
        cfg.dataset.edge_dim = 2
        cfg.dataset.edge_encoder_name = 'roland'
        cfg.dataset.node_encoder = True
        cfg.dataset.split = [0.7, 0.1, 0.2]
        cfg.transaction.snapshot_freq = 'D'

    elif name in ['bitcoinotc.csv', 'bitcoinalpha.csv']:
        cfg.dataset.format = 'bitcoin'
        cfg.dataset.edge_dim = 2
        cfg.dataset.edge_encoder_name = 'roland_general'
        cfg.dataset.node_encoder = False
        cfg.dataset.split = [0.7, 0.1, 0.2]
        cfg.transaction.snapshot_freq = '1200000s'
        if name == 'bitcoinotc.csv':
            cfg.train.start_compute_mrr = 110
        else:
            cfg.train.start_compute_mrr = 109

    elif name in ['CollegeMsg.txt']:
        cfg.dataset.format = 'uci_message'
        cfg.dataset.edge_dim = 1
        cfg.dataset.edge_encoder_name = 'roland_general'
        cfg.dataset.node_encoder = False
        cfg.dataset.split = [0.71, 0.09, 0.2]
        cfg.transaction.snapshot_freq = '190080s'
        cfg.train.start_compute_mrr = 72

    elif name in ['reddit-body.tsv', 'reddit-title.tsv']:
        cfg.dataset.format = 'reddit_hyperlink'
        cfg.dataset.edge_dim = 88
        cfg.dataset.edge_encoder_name = 'roland_general'
        cfg.dataset.node_encoder = False
        cfg.dataset.split = [0.7, 0.1, 0.2]
        cfg.transaction.snapshot_freq = 'D'

    elif name in ['AS-733']:
        cfg.dataset.format = 'as'
        cfg.dataset.edge_dim = 1
        cfg.dataset.edge_encoder_name = 'roland_general'
        cfg.dataset.node_encoder = False
        cfg.transaction.snapshot_freq = 'D'
        cfg.train.start_compute_mrr = 81

    else:
        raise ValueError(f'No default config for dataset {name}.')


def dataset_cfg_setup_live_update(name: str):
    """
    Setup required fields in cfg for the given dataset.
    """
    if name in ['bsi_svt_2008.tsv', 'bsi_zk_2008.tsv']:
        cfg.dataset.format = 'roland_bsi_general'
        cfg.dataset.edge_dim = 2
        cfg.dataset.edge_encoder_name = 'roland'
        cfg.dataset.node_encoder = True
        if name == 'bsi_svt_2008.tsv':
            cfg.transaction.feature_node_int_num = [1018, 33, 13, 23, 5]
            cfg.transaction.snapshot_freq = 'W'
        else:
            cfg.transaction.feature_node_int_num = [1483, 32, 13, 24, 5]
            cfg.transaction.snapshot_freq = 'D'

    elif name in ['bitcoinotc.csv', 'bitcoinalpha.csv']:
        cfg.dataset.format = 'bitcoin'
        cfg.dataset.edge_dim = 2
        cfg.dataset.edge_encoder_name = 'roland_general'
        cfg.dataset.node_encoder = False
        cfg.transaction.snapshot_freq = 'W'

    elif name in ['CollegeMsg.txt']:
        cfg.dataset.format = 'uci_message'
        cfg.dataset.edge_dim = 1
        cfg.dataset.edge_encoder_name = 'roland_general'
        cfg.dataset.node_encoder = False
        cfg.transaction.snapshot_freq = 'W'

#-----------------------RE/FRIEND-----------------------------#
    elif name in ['RE.txt']:
        cfg.dataset.format = 'RE_friend'
        cfg.dataset.edge_dim = 1
        cfg.dataset.edge_encoder_name = 'roland_general'
        cfg.dataset.node_encoder = False
        cfg.transaction.snapshot_freq = 'D'
    
    elif name in ['RE.txt']:
        cfg.dataset.format = 'RE'
        cfg.dataset.edge_dim = 1
        cfg.dataset.edge_encoder_name = 'roland_general'
        cfg.dataset.node_encoder = False
        cfg.transaction.snapshot_freq = 'D'
#----------------------------------------------------------#
#-----------------------RT/FRIEND-----------------------------#
    elif name in ['RT.txt']:
        cfg.dataset.format = 'RT_friend'
        cfg.dataset.edge_dim = 1
        cfg.dataset.edge_encoder_name = 'roland_general'
        cfg.dataset.node_encoder = False
        cfg.transaction.snapshot_freq = '28800s'
    
    elif name in ['RT.txt']:
        cfg.dataset.format = 'RT'
        cfg.dataset.edge_dim = 1
        cfg.dataset.edge_encoder_name = 'roland_general'
        cfg.dataset.node_encoder = False
        cfg.transaction.snapshot_freq = '28800s'
#----------------------------------------------------------#
#-----------------------MT/FRIEND-----------------------------#
    elif name in ['MT.txt']:
        cfg.dataset.format = 'MT_friend'
        cfg.dataset.edge_dim = 1
        cfg.dataset.edge_encoder_name = 'roland_general'
        cfg.dataset.node_encoder = False
        cfg.transaction.snapshot_freq = '43200s'
    
    elif name in ['MT.txt']:
        cfg.dataset.format = 'MT'
        cfg.dataset.edge_dim = 1
        cfg.dataset.edge_encoder_name = 'roland_general'
        cfg.dataset.node_encoder = False
        cfg.transaction.snapshot_freq = '43200s'
#----------------------------------------------------------#

    elif name in ['reddit-body.tsv', 'reddit-title.tsv']:
        cfg.dataset.format = 'reddit_hyperlink'
        cfg.dataset.edge_dim = 88
        cfg.dataset.edge_encoder_name = 'roland_general'
        cfg.dataset.node_encoder = False
        cfg.transaction.snapshot_freq = 'W'

    elif name in ['AS-733']:
        cfg.dataset.format = 'as'
        cfg.dataset.edge_dim = 1
        cfg.dataset.edge_encoder_name = 'roland_general'
        cfg.dataset.node_encoder = False
        cfg.transaction.snapshot_freq = 'D'

    else:
        raise ValueError(f'No default config for dataset {name}.')

def sample_negative_edges(snapshot_edge_index, sub_friend_edges, num_negative_samples):
    # Convert edge indices to sets for faster membership testing
    snapshot_edge_set = {(snapshot_edge_index[0][i].item(), snapshot_edge_index[1][i].item()) for i in range(snapshot_edge_index.size(1))}
    sub_friend_edge_set = {(sub_friend_edges[0][i].item(), sub_friend_edges[1][i].item()) for i in range(sub_friend_edges.size(1))}

    # Calculate the set difference to get edges that are in sub_friend_edges but not in snapshot_edge_index
    pre_sample = []
    for edge in sub_friend_edge_set:
        if edge not in snapshot_edge_set:
            pre_sample.append(edge)

    # Check if pre_sample has enough edges for negative sampling
    if len(pre_sample) >= num_negative_samples:
        # Randomly select num_negative_samples negative edges
        negative_samples = random.sample(pre_sample, num_negative_samples)
    else:
        # We need to introduce new edges until we have enough negative negative_samples
        num_missing_samples = num_negative_samples - len(pre_sample)
        candidate_edges = []

        # Generate candidate edges that do not exist in pre_sample or snapshot_edge_set
        while len(candidate_edges) < num_missing_samples:
            src = random.randint(0, snapshot_edge_index.size(1) - 1)
            dst = random.randint(0, snapshot_edge_index.size(1) - 1)
            candidate_edge = (snapshot_edge_index[0, src].item(), snapshot_edge_index[1, dst].item())

            if candidate_edge not in snapshot_edge_set and candidate_edge not in pre_sample:
                candidate_edges.append(candidate_edge)

        negative_samples = pre_sample + candidate_edges

    # Convert negative_samples to tensor
    negative_samples = torch.tensor(negative_samples, dtype=torch.long).t()

    return negative_samples

def build_network(filepath,datasets):
    with open(filepath, 'r') as file:
        sub_friend_content = file.readlines()

    num_nodes=datasets[0][0].node_feature.size(0)
    node_feature = torch.ones(num_nodes, 1)

    num_edges=len(sub_friend_content)
    edge_index=[]
    for line in sub_friend_content:
        src,dst=map(int,line.split())
        edge_index.append([src,dst])
    edge_index=torch.tensor(edge_index).t().long()
    edge_feature = torch.ones((num_edges, 1))

    graph = Graph(
        node_feature=node_feature,
        edge_feature=edge_feature,
        edge_index=edge_index,
        directed=True
    )

    NET_static=[]
    for i in range(len(datasets)):
        temp=[]
        for j in range(len(datasets[i])):
            snapshot=datasets[i][j]
            snapshot_nodes = set(snapshot.edge_index.view(-1).tolist())
            sub_friend_edges = [line.strip().split() for line in sub_friend_content \
                                if all(int(edge) in snapshot_nodes for edge in line.strip().split()[:2])]
            sub_friend_edges = [[int(edge[0]), int(edge[1])] for edge in sub_friend_edges]
            sub_friend_edges = torch.tensor(sub_friend_edges, dtype=torch.long).t()
            sub_num_edges=sub_friend_edges.size(1)

            sub_edge_feature = torch.ones((sub_num_edges, 1))

            ############################重构网络损失#####################################
            # Calculate the number of positive samples
            num_positive_samples = snapshot.edge_index.size(1)

            # Calculate the number of negative samples
            num_negative_samples = num_positive_samples

            negative_samples=sample_negative_edges(snapshot.edge_index,sub_friend_edges,num_negative_samples)
            
            # Stack positive and negative samples to create recon_index

            recon_index = torch.cat((snapshot.edge_index, negative_samples), dim=1)
            recon_label = torch.cat((torch.ones(num_positive_samples), torch.zeros(num_negative_samples)))

            sub_friend_graph = Graph(
            node_feature=graph.node_feature,
            edge_feature=sub_edge_feature,
            edge_index=sub_friend_edges,
            recon_index=recon_index,
            recon_label=recon_label,
            directed=True 
        )

            temp.append(sub_friend_graph)
        temp1=GraphDataset(graphs=temp)
        NET_static.append(temp1)
    return NET_static

if __name__ == '__main__':
    # Load cmd line args
    print("args:000")
    args = parse_args()
    print("args:",args)
    # Repeat for different random seeds
    for i in range(args.repeat):
        # Load config file
        cfg.merge_from_file(args.cfg_file)
        cfg.merge_from_list(args.opts)
        assert_cfg(cfg)

        # over-ride data path and remark if required.
        if args.override_data_dir is not None:
            cfg.dataset.dir = args.override_data_dir
        if args.override_remark is not None:
            cfg.remark = args.override_remark

        # Set Pytorch environment
        torch.set_num_threads(cfg.num_threads)
        out_dir_parent = cfg.out_dir
        cfg.seed = i + 1
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)
        update_out_dir(out_dir_parent, args.cfg_file)
        dump_cfg(cfg)
        setup_printing()
        auto_select_device()

        # Infer dataset format if required, retrieve the cfg setting associated
        # with the particular dataset.
        if cfg.dataset.format == 'infer':
            if cfg.train.mode in ['baseline', 'baseline_v2', 'live_update_fixed_split']:
                dataset_cfg_setup_fixed_split(cfg.dataset.name)
            elif cfg.train.mode == 'live_update':
                dataset_cfg_setup_live_update(cfg.dataset.name)
            elif cfg.train.mode == 'live_update_baseline':
                dataset_cfg_setup_live_update(cfg.dataset.name)

        # Set learning environment
        if cfg.dataset.premade_datasets in ['fresh', 'fresh_save_cache']:
            '''cfg.dataset.format = 'RE_friend'
            datasets1 = create_dataset()'''
            cfg.dataset.format='RE'
            datasets= create_dataset()
        else:
            datasets = torch.load(os.path.join(cfg.dataset.dir,
                                               cfg.dataset.premade_datasets))
        
        ################load静态层网络###################
        filepath = f'/home/ly/SDNL/Bitcoinalpha_T/sub_friend.txt'
        NET_static=build_network(filepath,datasets)
        
        ###############看一下划分结果###############
        '''for idx in range(len(datasets[0])):
            snapshot=datasets[0][idx]
            snapshot1=datasets[1][idx]
            snapshot2=datasets[2][idx]'''
        '''snapshot_file_path = f'/root/autodl-tmp/demo/RE_friend/split/RE0_{idx}.txt'
            snapshot_file_path1 = f'/root/autodl-tmp/demo/RE_friend/split/RE1_{idx}.txt'
            snapshot_file_path2 = f'/root/autodl-tmp/demo/RE_friend/split/RE2_{idx}.txt'
            with open(snapshot_file_path, 'w') as snapshot_file:
                for edge in snapshot.edge_index.t().tolist():
                    snapshot_file.write(f"{edge[0]} {edge[1]}\n")
            with open(snapshot_file_path1, 'w') as snapshot_file1:
                for edge in snapshot1.edge_index.t().tolist():
                    snapshot_file1.write(f"{edge[0]} {edge[1]}\n")
            with open(snapshot_file_path2, 'w') as snapshot_file2:
                for edge in snapshot2.edge_index.t().tolist():
                    snapshot_file2.write(f"{edge[0]} {edge[1]}\n")'''  

        '''snapshot_file_path = f'/root/autodl-tmp/demo/RE_friend/split/label0_{idx}.txt'
            snapshot_file_path1 = f'/root/autodl-tmp/demo/RE_friend/split/label1_{idx}.txt'
            snapshot_file_path2 = f'/root/autodl-tmp/demo/RE_friend/split/label2_{idx}.txt'
            with open(snapshot_file_path, 'w') as snapshot_file:
                for edge in snapshot.edge_label.t().tolist():
                    snapshot_file.write(f"{edge}\n")
            with open(snapshot_file_path1, 'w') as snapshot_file1:
                for edge in snapshot1.edge_label.t().tolist():
                    snapshot_file1.write(f"{edge}\n")
            with open(snapshot_file_path2, 'w') as snapshot_file2:
                for edge in snapshot2.edge_label.t().tolist():
                    snapshot_file2.write(f"{edge}\n")'''  
               
        # required by gconv-lstm-h
        cfg.dataset.num_nodes = datasets[0][0].num_nodes
        cfg.model.type='gnn_recurrent_loss'
        loaders = create_loader(datasets)
        meters = create_logger(datasets, loaders)
        model = create_model(datasets)
        optimizer = create_optimizer(model.parameters())
        scheduler = create_scheduler(optimizer)

        '''cfg.dataset.edge_dim =1
        # ----------------------------------------------------------#
        loaders1 = create_loader(datasets1)
        meters1 = create_logger(datasets1, loaders1)
        model1 = create_model(datasets1)
        optimizer1 = create_optimizer(model1.parameters())
        scheduler1 = create_scheduler(optimizer1)'''

        # Print model info
        logging.info(model)
        logging.info(cfg)
        cfg.params = params_count(model)
        logging.info('Num parameters: {}'.format(cfg.params))
        
        # Start training
        for dataset, name in zip(datasets, ('train', 'validation', 'test')):
            print(f'{name} set: {len(dataset)} graphs.')
            '''all_edge_time = torch.cat([g.edge_time for g in dataset])
            start = int(torch.min(all_edge_time))
            start = datetime.fromtimestamp(start)
            end = int(torch.max(all_edge_time))
            end = datetime.fromtimestamp(end)
            print(f'\tRange: {start} - {end}')'''

        if cfg.dataset.negative_sample_weight != 'uniform':
            for dataset in datasets:
                dataset.negative_sample_weight = cfg.dataset.negative_sample_weight
        # breakpoint()
        if cfg.train.mode == 'standard':
            train(meters, loaders, model, optimizer, scheduler)
        else:
            train_dict[cfg.train.mode](
                meters, loaders, model, optimizer, scheduler, datasets,NET_static)

    if cfg.train.mode not in ['live_update', 'live_update_baseline', 'live_update_fixed_split']:
        # Don't need to involve loggers in the live update scheme.
        # Aggregate results from different seeds
        agg_runs(get_parent_dir(out_dir_parent, args.cfg_file), cfg.metric_best)
    # When being launched in batch mode, mark a yaml as done
    if args.mark_done:
        os.rename(args.cfg_file, '{}_done'.format(args.cfg_file))