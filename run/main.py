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

    elif name in ['CollegeMsg.txt']:
        cfg.dataset.format = 'uci_message'
        cfg.dataset.edge_dim = 1
        cfg.dataset.edge_encoder_name = 'roland_general'
        cfg.dataset.node_encoder = False
        cfg.dataset.split = [0.71, 0.09, 0.2]
        cfg.transaction.snapshot_freq = '190080s'
        cfg.train.start_compute_mrr = 72
    
    elif name in ['cannes.txt']:
        cfg.dataset.format = 'multi'
        cfg.dataset.edge_dim = 1
        cfg.dataset.edge_encoder_name = 'roland_general'
        cfg.dataset.node_encoder = False
        cfg.transaction.snapshot_freq = '14400s'
        cfg.dataset.split = [0.7, 0.1, 0.2]
        cfg.train.start_compute_mrr = 72
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
        cfg.transaction.snapshot_freq = '12M'

    elif name in ['CollegeMsg.txt']:
        cfg.dataset.format = 'uci_message'
        cfg.dataset.edge_dim = 1
        cfg.dataset.edge_encoder_name = 'roland_general'
        cfg.dataset.node_encoder = False
        cfg.transaction.snapshot_freq = 'M'

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
    
    elif name in ['cannes.txt']:
        cfg.dataset.format = 'multi'
        cfg.dataset.edge_dim = 1
        cfg.dataset.edge_encoder_name = 'roland_general'
        cfg.dataset.node_encoder = False
        cfg.transaction.snapshot_freq = '10800s'

    elif name in ['higgs.txt']:
        cfg.dataset.format = 'multi'
        cfg.dataset.edge_dim = 1
        cfg.dataset.edge_encoder_name = 'roland_general'
        cfg.dataset.node_encoder = False
        cfg.transaction.snapshot_freq = '10800s'

    elif name in ['mlking.txt']:
        cfg.dataset.format = 'multi'
        cfg.dataset.edge_dim = 1
        cfg.dataset.edge_encoder_name = 'roland_general'
        cfg.dataset.node_encoder = False
        cfg.transaction.snapshot_freq = '10800s'

    elif name in ['moscow.txt']:
        cfg.dataset.format = 'multi'
        cfg.dataset.edge_dim = 1
        cfg.dataset.edge_encoder_name = 'roland_general'
        cfg.dataset.node_encoder = False
        cfg.transaction.snapshot_freq = '10800s'

    elif name in ['nyclimate.txt']:
        cfg.dataset.format = 'multi'
        cfg.dataset.edge_dim = 1
        cfg.dataset.edge_encoder_name = 'roland_general'
        cfg.dataset.node_encoder = False
        cfg.transaction.snapshot_freq = '10800s'

    else:
        raise ValueError(f'No default config for dataset {name}.')

def select_edge(edges, num_samples):
    # 创建字典来存储节点的出度和入度
    out_degrees = {}
    in_degrees = {}

    for edge in edges:
        source, target = edge

        # 计算出度
        if source in out_degrees:
            out_degrees[source] += 1
        else:
            out_degrees[source] = 1

        # 计算入度
        if target in in_degrees:
            in_degrees[target] += 1
        else:
            in_degrees[target] = 1

    # 计算每条边的源节点出度和目标节点入度之和
    edge_scores = {}
    for edge in edges:
        source, target = edge
        score = out_degrees.get(source, 0) + in_degrees.get(target, 0)
        edge_scores[edge] = score

    # 根据得分对边进行排序，升序排序
    sorted_edges = sorted(edge_scores.keys(), key=lambda edge: edge_scores[edge])

    # 选择前 num_samples 条边
    # selected_edges = sorted_edges[:num_samples]
    selected_edges = random.sample(sorted_edges, num_samples)

    return selected_edges

def select_random_edges(edges, num_samples):
    return random.sample(edges, num_samples)

def sample_negative_edges(snapshot_edge_index, sub_friend_edges):
    # Convert edge indices to sets for faster membership testing
    snapshot_edge_set = {(snapshot_edge_index[0][i].item(), snapshot_edge_index[1][i].item()) for i in range(snapshot_edge_index.size(1))}
    sub_friend_edge_set = {(sub_friend_edges[0][i].item(), sub_friend_edges[1][i].item()) for i in range(sub_friend_edges.size(1))}

    dy_set = snapshot_edge_set - sub_friend_edge_set
    st_set = sub_friend_edge_set - snapshot_edge_set

    if len(st_set)>=len(dy_set):
        num_single_samples = len(snapshot_edge_set)
        sample_edge = select_edge(st_set,num_single_samples)
        # sample_edge = select_random_edges(st_set,num_single_samples)
        positive_samples = torch.tensor(list(snapshot_edge_set), dtype=torch.long).t()
        negative_samples = torch.tensor(sample_edge, dtype=torch.long).t()

    else:
        num_single_samples = len(sub_friend_edge_set)
        sample_edge = select_edge(dy_set,num_single_samples)
        # sample_edge = select_random_edges(st_set,num_single_samples)
        positive_samples = torch.tensor(sample_edge, dtype=torch.long).t()
        negative_samples = torch.tensor(list(sub_friend_edge_set), dtype=torch.long).t()    

    return positive_samples, negative_samples, num_single_samples

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
        torch.cuda.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cudnn.deterministic = True  # 确定性算法
        torch.backends.cudnn.benchmark = False # 禁用CuDNN的自动优化算法选择
        update_out_dir(out_dir_parent, args.cfg_file)
        dump_cfg(cfg)
        setup_printing()
        auto_select_device()
        # Infer dataset format if required, retrieve the cfg setting associated
        # with the particular dataset.

        if cfg.dataset.format == 'infer':
            print('看看训练模式是什么？',cfg.train.mode)
            if cfg.train.mode in ['baseline', 'baseline_v2', 'live_update_fixed_split']:
                dataset_cfg_setup_fixed_split(cfg.dataset.name)
            elif cfg.train.mode == 'live_update':
                dataset_cfg_setup_live_update(cfg.dataset.name)

            elif cfg.train.mode == 'live_update_baseline':
                dataset_cfg_setup_live_update(cfg.dataset.name)

        # Set learning environment
        if cfg.dataset.premade_datasets in ['fresh', 'fresh_save_cache']:
            cfg.dataset.format='multi'
            datasets1, datasets2, datasets3 = create_dataset()
        else:
            datasets = torch.load(os.path.join(cfg.dataset.dir,
                                               cfg.dataset.premade_datasets))

        cfg.model.type='gnn_recurrent_multi'

        loaders_list = list()
        meters_list = list()
        datasets_list = list()

        loaders1 = create_loader(datasets1)
        meters1 = create_logger(datasets1, loaders1)
        loaders2 = create_loader(datasets2)
        meters2 = create_logger(datasets2, loaders2)
        loaders3 = create_loader(datasets3)
        meters3 = create_logger(datasets3, loaders3)

        loaders_list=[loaders1,loaders2,loaders3]
        meters_list=[meters1,meters2,meters3]
        datasets_list=[datasets1, datasets2, datasets3]

        model = create_model(datasets1,datasets2,datasets3)
        optimizer = create_optimizer(model.parameters())
        scheduler = create_scheduler(optimizer)

        # Print model info
        logging.info(model)
        logging.info(cfg)
        cfg.params = params_count(model)
        logging.info('Num parameters: {}'.format(cfg.params))
        
        # Start training
        for dataset1, name in zip(datasets1, ('train', 'validation', 'test')):
            print(f'{name} set: {len(dataset1)} graphs.')
            
        for dataset2, name in zip(datasets2, ('train', 'validation', 'test')):
            print(f'{name} set: {len(dataset2)} graphs.')

        for dataset3, name in zip(datasets3, ('train', 'validation', 'test')):
            print(f'{name} set: {len(dataset3)} graphs.')

        if cfg.dataset.negative_sample_weight != 'uniform':
            for dataset in datasets:
                dataset.negative_sample_weight = cfg.dataset.negative_sample_weight
        # breakpoint()
        if cfg.train.mode == 'standard':
            train(meters, loaders, model, optimizer, scheduler)
        else:
            train_dict[cfg.train.mode](
                meters_list, loaders_list, model, optimizer, scheduler, datasets_list)

    if cfg.train.mode not in ['live_update', 'live_update_baseline', 'live_update_fixed_split']:
        # Don't need to involve loggers in the live update scheme.
        # Aggregate results from different seeds
        agg_runs(get_parent_dir(out_dir_parent, args.cfg_file), cfg.metric_best)
    # When being launched in batch mode, mark a yaml as done
    if args.mark_done:
        os.rename(args.cfg_file, '{}_done'.format(args.cfg_file))