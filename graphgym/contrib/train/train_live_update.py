"""
The more realistic training pipeline.
"""
import copy
import datetime
import json
import logging
import os
from typing import Dict, List, Optional, Sequence, Tuple
from sklearn.metrics import roc_auc_score, average_precision_score

import deepsnap
import numpy as np
import pandas as pd

import torch
from graphgym.checkpoint import clean_ckpt
from graphgym.config import cfg
from graphgym.contrib.train import train_utils
from graphgym.loss import compute_loss
from graphgym.optimizer import create_optimizer, create_scheduler
from graphgym.register import register_train
from graphgym.utils.io import makedirs_rm_exist
from graphgym.utils.stats import node_degree
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

@torch.no_grad()
def average_state_dict(dict1: dict, dict2: dict, weight: float) -> dict:
    # Average two model.state_dict() objects.
    # out = (1-w)*dict1 + w*dict2
    assert 0 <= weight <= 1
    d1 = copy.deepcopy(dict1)
    d2 = copy.deepcopy(dict2)
    out = dict()
    for key in d1.keys():
        assert isinstance(d1[key], torch.Tensor)
        param1 = d1[key].detach().clone()
        assert isinstance(d2[key], torch.Tensor)
        param2 = d2[key].detach().clone()
        out[key] = (1 - weight) * param1 + weight * param2
    return out


def _dataset_to_list(dataset) -> list:
    return [dataset[i] for i in range(len(dataset))]


def _merge_temporal_splits_if_needed(datasets, net_static):
    """Use one full timeline; split labels inside each snapshot instead."""
    if cfg.dataset.split_method != 'chronological_temporal':
        return datasets, net_static
    if len(datasets) != 3:
        return datasets, net_static
    if datasets[0] is datasets[1] and datasets[1] is datasets[2]:
        return datasets, net_static

    merged_dataset = []
    merged_static = []
    for split_dataset, split_static in zip(datasets, net_static):
        merged_dataset.extend(_dataset_to_list(split_dataset))
        merged_static.extend(_dataset_to_list(split_static))
    return [merged_dataset, merged_dataset, merged_dataset], \
        [merged_static, merged_static, merged_static]


def _split_counts(num_items: int, ratios: Sequence[float]) -> Tuple[int, int, int]:
    if len(ratios) != 3:
        raise ValueError(f'cfg.dataset.split must have 3 values, got {ratios}.')
    if num_items <= 0:
        return 0, 0, 0

    ratio_sum = float(sum(ratios))
    if ratio_sum <= 0:
        raise ValueError(f'cfg.dataset.split must sum to a positive value, got {ratios}.')
    ratios = [float(x) / ratio_sum for x in ratios]

    train_count = int(num_items * ratios[0])
    val_count = int(num_items * ratios[1])
    test_count = num_items - train_count - val_count

    if num_items >= 3:
        if val_count == 0:
            val_count = 1
        if test_count == 0:
            test_count = 1
        train_count = num_items - val_count - test_count
    return train_count, val_count, test_count


def _split_indices(num_items: int, ratios: Sequence[float]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    perm = torch.randperm(num_items)
    train_count, val_count, test_count = _split_counts(num_items, ratios)
    train_idx = perm[:train_count]
    val_idx = perm[train_count:train_count + val_count]
    test_idx = perm[train_count + val_count:train_count + val_count + test_count]
    return train_idx, val_idx, test_idx


def _sample_negative_edges(positive_edge_index: torch.Tensor,
                           num_nodes: int,
                           num_samples: int) -> torch.Tensor:
    if num_samples == 0:
        return torch.empty((2, 0), dtype=torch.long)

    pos_cpu = positive_edge_index.detach().cpu().long()
    positive_edges = set(map(tuple, pos_cpu.t().tolist()))
    negative_edges = set()
    max_possible = num_nodes * max(num_nodes - 1, 0) - len(positive_edges)
    if max_possible < num_samples:
        raise ValueError(
            f'Cannot sample {num_samples} negative edges from {num_nodes} nodes '
            f'with {len(positive_edges)} positive edges.'
        )

    attempts = 0
    while len(negative_edges) < num_samples:
        remaining = num_samples - len(negative_edges)
        sample_size = max(remaining * 2, 1024)
        src = torch.randint(0, num_nodes, (sample_size,), dtype=torch.long)
        dst = torch.randint(0, num_nodes, (sample_size,), dtype=torch.long)
        for edge in zip(src.tolist(), dst.tolist()):
            if edge[0] == edge[1]:
                continue
            if edge in positive_edges or edge in negative_edges:
                continue
            negative_edges.add(edge)
            if len(negative_edges) == num_samples:
                break
        attempts += 1
        if attempts > 1000:
            raise RuntimeError('Negative sampling did not converge; graph may be too dense.')

    return torch.tensor(list(negative_edges), dtype=torch.long).t().contiguous()


@torch.no_grad()
def build_edge_label_splits(dataset, task: Tuple[int, int]) -> List[dict]:
    """Construct per-snapshot train/val/test labels from dynamic edges."""
    _, tomorrow = task
    target_graph = dataset[tomorrow]
    positive_edges = target_graph.edge_index.detach().cpu().long()
    num_pos = positive_edges.shape[1]
    num_nodes = int(target_graph.node_feature.shape[0])
    negative_edges = _sample_negative_edges(positive_edges, num_nodes, num_pos)

    pos_splits = _split_indices(num_pos, cfg.dataset.split)
    neg_splits = _split_indices(num_pos, cfg.dataset.split)

    label_splits = []
    for pos_idx, neg_idx in zip(pos_splits, neg_splits):
        pos_edges = positive_edges[:, pos_idx]
        neg_edges = negative_edges[:, neg_idx]
        edge_label_index = torch.cat((pos_edges, neg_edges), dim=1).contiguous()
        edge_label = torch.cat((
            torch.ones(pos_edges.shape[1]),
            torch.zeros(neg_edges.shape[1])
        ), dim=0)

        if edge_label.numel() > 0:
            perm = torch.randperm(edge_label.numel())
            edge_label_index = edge_label_index[:, perm]
            edge_label = edge_label[perm]
        label_splits.append({
            'edge_label_index': edge_label_index,
            'edge_label': edge_label
        })
    return label_splits

def _get_live_update_val_score(perf: dict) -> Tuple[float, str]:
    """Return Algorithm 1's validation criterion Met_val."""
    metric = str(getattr(cfg, 'metric_best', 'auto')).lower()
    metric_map = {
        'auto': 'micro_auc',
        'auc': 'micro_auc',
        'micro_auc': 'micro_auc',
        'macro_auc': 'macro_auc',
        'ap': 'ap',
        'average_precision': 'ap',
        'acc': 'accuracy',
        'accuracy': 'accuracy',
    }
    if metric in ['loss', 'val_loss']:
        return -float(perf['loss']), 'neg_loss'
    perf_key = metric_map.get(metric, metric)
    if perf_key not in perf:
        raise ValueError(
            f"Unsupported cfg.metric_best={cfg.metric_best!r} for live_update. "
            "Use auto/micro_auc/macro_auc/ap/accuracy/loss.")
    return float(perf[perf_key]), perf_key


def precompute_edge_degree_info(dataset):
    """Pre-computes edge_degree_existing, edge_degree_new and keep ratio
    at each snapshot. Inplace modifications.
    """
    num_nodes = dataset[0].node_feature.shape[0]

    count = torch.zeros(num_nodes)
    act = torch.zeros_like(count, dtype=torch.bool)
    act1 = torch.zeros_like(act, dtype=torch.bool)

    for t in tqdm(range(len(dataset)), desc='precompute edge deg info'):
        if t == 0:
            dataset[t].node_degree_existing = torch.zeros(num_nodes)
        else:
            dataset[t].node_degree_existing \
                = dataset[t - 1].node_degree_existing \
                  + dataset[t - 1].node_degree_new

        dataset[t].node_degree_new = node_degree(dataset[t].edge_index,
                                                 n=num_nodes)
        act = act | (dataset[t].node_degree_new != 0).bool()
        act1 = (dataset[t].node_degree_new != 0).bool()
        
        # 判断 act 和 act1 对应元素是否相等
        count_update = torch.where(act == act1, torch.zeros_like(count), count + 1)
        # 更新 count
        count.copy_(count_update)
 
        dataset[t].keep_ratio = train_utils.get_keep_ratio(
            existing=dataset[t].node_degree_existing,
            new=dataset[t].node_degree_new,
            mode=cfg.transaction.keep_ratio)
        
        # 将tensor([0,0,0……,0])转化为tensor([[0],[0]……,[0]])，此外keep_ratio中只有小于1的值
        dataset[t].keep_ratio = dataset[t].keep_ratio.unsqueeze(-1)

# @torch.no_grad()
# def get_task(dataset: deepsnap.dataset.GraphDataset,
#              today: int, tomorrow: int) -> deepsnap.graph.Graph:
#     """
#     Construct batch required for the task (today, tomorrow). As defined in
#     batch's get_item method (used to get edge_label and get_label_index),
#     edge_label and edge_label_index returned would be different everytime
#     get_task() is called.
#     """
#     assert today < tomorrow < len(dataset)
#     batch = dataset[today].clone()
#     batch.edge_label = dataset[tomorrow].edge_label.clone()
#     batch.edge_label_index = dataset[tomorrow].edge_label_index.clone()
#
#     batch = train_utils.move_batch_to_device(batch, cfg.device)
#     return batch


@torch.no_grad()
def get_task_batch(dataset: deepsnap.dataset.GraphDataset,
                   today: int, tomorrow: int,
                   prev_node_states: Optional[Dict[str, List[torch.Tensor]]],
                   label_split: Optional[dict] = None
                   ) -> deepsnap.graph.Graph:
    """
    Construct batch required for the task (today, tomorrow). As defined in
    batch's get_item method (used to get edge_label and get_label_index),
    edge_label and edge_label_index returned would be different everytime
    get_task_batch() is called.

    Moreover, copy node-memories (node_states and node_cells) to the batch.
    """

    # assert today < tomorrow < len(dataset)
    # Get edges for message passing and prediction task.
    batch = dataset[today].clone()
    if label_split is None:
        batch.edge_label = dataset[tomorrow].edge_label.clone()
        batch.edge_label_index = dataset[tomorrow].edge_label_index.clone()
    else:
        batch.edge_label = label_split['edge_label'].clone()
        batch.edge_label_index = label_split['edge_label_index'].clone()

    # Copy previous memory to the batch.
    if prev_node_states is not None:
        for key, val in prev_node_states.items():
            copied = [x.detach().clone() for x in val]
            setattr(batch, key, copied)

    batch = train_utils.move_batch_to_device(batch, cfg.device)
    return batch

@torch.no_grad()
def get_task_batch_dy(dataset: deepsnap.dataset.GraphDataset,
                   today: int, tomorrow: int,
                   prev_node_states: Optional[Dict[str, List[torch.Tensor]]],prev_node_states1: Optional[Dict[str, List[torch.Tensor]]]
                   ) -> deepsnap.graph.Graph:
    """
    Construct batch required for the task (today, tomorrow). As defined in
    batch's get_item method (used to get edge_label and get_label_index),
    edge_label and edge_label_index returned would be different everytime
    get_task_batch() is called.

    Moreover, copy node-memories (node_states and node_cells) to the batch.
    """

    assert today < tomorrow < len(dataset)
    # Get edges for message passing and prediction task.
    batch = dataset[today].clone()
    batch.edge_label = dataset[tomorrow].edge_label.clone()
    batch.edge_label_index = dataset[tomorrow].edge_label_index.clone()

    # Copy previous memory to the batch.
    if prev_node_states is not None:
        for key, val in prev_node_states.items():
            copied = [x.detach().clone() for x in val]
            setattr(batch, key, copied)

    if prev_node_states1 is not None:
        new_key_mapping = {
            'node_states': 'node_states1',
            'node_cells':'node_cells1'
        # 添加其他键的映射关系...
        }

        for original_key, val in prev_node_states1.items():
            new_key = new_key_mapping.get(original_key, original_key)  # 如果映射不存在，保持原来的键名
            copied = [x.detach().clone() for x in val]
            setattr(batch, new_key, copied)
    else:
            node_states_copy = copy.deepcopy(batch.node_states)
            setattr(batch,'node_states1',node_states_copy)
    batch = train_utils.move_batch_to_device(batch, cfg.device)
    return batch

@torch.no_grad()
def update_node_states(model, dataset,net_static,task: Tuple[int, int],
                       prev_node_states: Optional[
                           Dict[str, List[torch.Tensor]]],
                       label_split: Optional[dict] = None
                       ) -> Dict[str, List[torch.Tensor]]:
    """Perform the provided task and keep track of the latest node_states.

    Example: task = (t, t+1),
        the prev_node_states contains node embeddings at time (t-1).
        the model perform task (t, t+1):
            Input: (node embedding at t - 1, edges at t).
            Output: possible transactions at t+1.
        the model also generates node embeddings at t.

    after doing task (t, t+1), node_states contains information
    from snapshot t.
    """
    today, tomorrow = task
    batch = get_task_batch(dataset, today, tomorrow, prev_node_states,
                           label_split).clone()
    batch1= net_static[today].clone()
    batch1 = batch1.to(device=cfg.device)
    # Let the model modify batch.node_states (and batch.node_cells).
    _, _ = model(batch,batch1)
    # Collect the updated node states.
    out = dict()
    out['node_states'] = [x.detach().clone() for x in batch.node_states]
    if isinstance(batch.node_cells[0], torch.Tensor):
        out['node_cells'] = [x.detach().clone() for x in batch.node_cells]

    return out

def train_step(model, optimizer, scheduler, dataset,net_static,
               task: Tuple[int, int],
               prev_node_states: Optional[Dict[str, torch.Tensor]],
               label_split: Optional[dict] = None
               ) -> dict:
    """
    After receiving ground truth from a particular task, update the model by
    performing back-propagation.
    For example, on day t, the ground truth of task (t-1, t) has been revealed,
    train the model using G[t-1] for message passing and label[t] as target.
    """
    optimizer.zero_grad()
    torch.cuda.empty_cache()

    today, tomorrow = task
    model.train()
    batch = get_task_batch(dataset, today, tomorrow, prev_node_states,
                           label_split).clone()
    batch1= net_static[today].clone()
    batch1 = batch1.to(device=cfg.device)

    pred, true = model(batch,batch1)
    loss, pred_score = compute_loss(pred, true)
    loss.backward()
    optimizer.step()

    scheduler.step()
    return {'loss': loss}

@torch.no_grad()
def evaluate_step(model, dataset,net_static,task: Tuple[int, int],
                  prev_node_states: Optional[Dict[str, List[torch.Tensor]]],
                  label_split: Optional[dict] = None,
                  fast: bool = False) -> dict:
    """
    Evaluate model's performance on task = (today, tomorrow)
        where today and tomorrow are integers indexing snapshots.
    """
    today, tomorrow = task
    model.eval()
    batch = get_task_batch(dataset, today, tomorrow, prev_node_states,
                           label_split).clone()
    batch1= net_static[today].clone()
    batch1 = batch1.to(device=cfg.device)
    pred, true = model(batch,batch1)
    loss, pred_score = compute_loss(pred, true)

    micro_auc = roc_auc_score(y_true=true.detach().cpu().numpy(), y_score=pred_score.detach().cpu().numpy(), average='micro')
    macro_auc = roc_auc_score(y_true=true.detach().cpu().numpy(), y_score=pred_score.detach().cpu().numpy(), average='macro')
    ap = average_precision_score(y_true=true.detach().cpu().numpy(), y_score=pred_score.detach().cpu().numpy())

    pred_label = torch.zeros(len(pred_score))
    pred_label[pred_score >= 0.5] = 1.0
    accuracy = np.mean(true.detach().cpu().numpy() == pred_label.numpy())
    if fast:
        # skip MRR calculation for internal validation.
        return {'loss': loss.item(), 'micro_auc': micro_auc, 'macro_auc': macro_auc, 'accuracy': accuracy,'ap':ap}

    mrr_batch = get_task_batch(dataset, today, tomorrow,
                               prev_node_states, label_split).clone()
    batch1= net_static[today].clone()
    batch1 = batch1.to(device=cfg.device)

    mrr, rck1, rck3, rck10 = train_utils.report_rank_based_eval(
        mrr_batch,batch1,model,
        num_neg_per_node=cfg.experimental.rank_eval_multiplier)

    return {'loss': loss.item(), 'mrr': mrr, 'rck1': rck1, 'rck3': rck3,
            'rck10': rck10, 'micro_auc': micro_auc, 'macro_auc': macro_auc, 'accuracy': accuracy,'ap':ap}

def train_live_update(loggers, loaders, model, optimizer, scheduler, datasets,NET_static,
                      **kwargs):
    datasets, NET_static = _merge_temporal_splits_if_needed(datasets, NET_static)

    for dataset in datasets:
        # Sometimes edge degree info is already included in dataset.
        if not hasattr(dataset[0], 'keep_ratio'):
            precompute_edge_degree_info(dataset)

    if cfg.dataset.premade_datasets == 'fresh_save_cache':
        if not os.path.exists(f'{cfg.dataset.dir}/cache/'):
            os.mkdir(f'{cfg.dataset.dir}/cache/')
        cache_path = '{}/cache/cached_datasets_{}_{}_{}.pt'.format(
            cfg.dataset.dir, cfg.dataset.format.replace('.tsv', ''),
            cfg.transaction.snapshot_freq,
            datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
        )
        torch.save(datasets, cache_path)

    num_splits = len(loggers)  # train/val/test splits.
    # range for today in (today, tomorrow) task pairs.
    task_range = range(len(datasets[0]) - cfg.transaction.horizon)

    t = datetime.datetime.now().strftime('%b%d_%H-%M-%S')

    # directory to store tensorboard files of this run.
    out_dir = cfg.out_dir.replace('/', '\\')
    # dir to store all run outputs for the entire batch.
    run_dir = 'runs_' + cfg.remark

    print(f'Tensorboard directory: {out_dir}')
    # If tensorboard directory exists, this config is in the re-run phase
    # of run_batch, replace logs of previous runs with the new one.
    makedirs_rm_exist(f'./{run_dir}/{out_dir}')
    writer = SummaryWriter(f'./{run_dir}/{out_dir}')

    # save a copy of configuration for later identifications.
    with open(f'./{run_dir}/{out_dir}/config.yaml', 'w') as f:
        cfg.dump(stream=f)

# ~~~~~~~~~~~~~~~~~~~~~~~~~核心部分~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    prev_node_states = None  # no previous state on day 0.
    # {'node_states': [Tensor, Tensor], 'node_cells: [Tensor, Tensor]}

    model_init = None  # for meta-learning only, a model.state_dict() object.

    auc_hist = list()
    mrr_hist = list()
    acc_hist = list()
    rck1_hist = list()
    rck3_hist = list()
    rck10_hist = list()
    ap_hist = list()
    loss_hist = list()
    internal_loss_records = list()

    for t in tqdm(task_range, desc='snapshot', leave=True):
        # current task: t --> t+1.
        label_splits = build_edge_label_splits(datasets[0], (t, t + 1))
        # (1) Evaluate model's performance on this task, at this time, the
        # model has seen no information on t+1, this evaluation is fair.
        for i in range(1, num_splits):
            perf = evaluate_step(model, datasets[i], NET_static[i], (t, t + 1),
                                 prev_node_states, label_splits[i])
            split_name = 'val' if i == 1 else 'test'
            tqdm.write(
                '[snapshot {}->{}][{}] loss={:.4f} auc={:.4f} '
                'acc={:.4f} ap={:.4f}'.format(
                    t, t + 1, split_name, float(perf['loss']),
                    float(perf['micro_auc']), float(perf['accuracy']),
                    float(perf['ap'])))
            
            if i == 2:
                loss_hist.append(perf['loss'])
                auc_hist.append(perf['micro_auc'])
                mrr_hist.append(perf['mrr'])
                acc_hist.append(perf['accuracy'])
                rck1_hist.append(perf['rck1'])
                rck3_hist.append(perf['rck3'])
                rck10_hist.append(perf['rck10'])
                ap_hist.append(perf['ap'])
            writer.add_scalars('val' if i == 1 else 'test', perf, t)

        # ————————————————————动态层—————————————————————————————————————— #
        if t <= cfg.train.stop_live_update_after:
            # (2) Reveal the ground truth of task (t, t+1) and update the model
            # to prepare for the next task.
            del optimizer, scheduler  # use new optimizers.

            optimizer = create_optimizer(model.parameters())
            scheduler = create_scheduler(optimizer)

            # best model's validation metric, training epochs, and state_dict.
            # Paper Algorithm 1 selects by Met_val increasing, not val_loss decreasing.
            best_model = {
                'val_score': -np.inf,
                'val_metric': None,
                'val_loss': np.inf,
                'train_epoch': 0,
                'state': None,
            }

            # keep track of how long we have NOT update the best model.
            best_model_unchanged = 0

            # after not updating the best model for `tol` epochs, stop.
            tol = cfg.train.internal_validation_tolerance

            # internal training loop (intra-snapshot cross-validation).
            # choose the best model using current validation set, prepare for
            # next task.

            if cfg.meta.is_meta and (model_init is not None):
                # For meta-learning, start fine-tuning from the pre-computed
                # initialization weight.
                model.load_state_dict(copy.deepcopy(model_init))         

            last_train_loss = None
            val_loss_trace = []
            train_loss_trace = []
            for i in tqdm(range(cfg.optim.max_epoch + 1), desc='live update',
                        leave=True):
                # Start with the un-trained model (i = 0), evaluate the model.

                internal_val_perf = evaluate_step(model, datasets[1], NET_static[1],
                                                (t, t + 1),
                                                prev_node_states, label_splits[1],
                                                fast=True)

                val_loss = internal_val_perf['loss']
                val_score, val_metric = _get_live_update_val_score(
                    internal_val_perf)
                val_loss_trace.append(float(val_loss))
                writer.add_scalar(
                    f'internal/snapshot_{t}_val_loss', float(val_loss), i)
                writer.add_scalar(
                    f'internal/snapshot_{t}_{val_metric}', float(val_score), i)

                if val_score > best_model['val_score']:
                    # replace the best model with the current model.
                    best_model = {
                        'val_score': val_score,
                        'val_metric': val_metric,
                        'val_loss': val_loss,
                        'train_epoch': i,
                        'state': copy.deepcopy(model.state_dict()),
                    }
                    best_model_unchanged = 0
                else:
                    # the current best model has dominated for these epochs.
                    best_model_unchanged += 1

                # if (i >= 2 * tol) and (best_model_unchanged >= tol):
                if best_model_unchanged >= tol:
                    # If the best model has not been updated for a while, stop.
                    break
                else:
                    # Otherwise, keep training.
                    train_perf = train_step(model, optimizer, scheduler,
                                            datasets[0], NET_static[0],
                                            (t, t + 1), prev_node_states,
                                            label_splits[0])
                    last_train_loss = float(train_perf['loss'].detach().cpu())
                    train_loss_trace.append(last_train_loss)
                    writer.add_scalar(
                        f'internal/snapshot_{t}_train_loss', last_train_loss, i)
                    writer.add_scalars('train', train_perf, t)
            

            writer.add_scalar('internal_best_val', best_model['val_score'], t)
            writer.add_scalar('internal_best_val_loss', best_model['val_loss'], t)
            writer.add_scalar('best epoch', best_model['train_epoch'], t)
            train_loss_msg = (
                '{:.4f}'.format(last_train_loss)
                if last_train_loss is not None else 'nan'
            )
            for epoch_idx, epoch_val_loss in enumerate(val_loss_trace):
                train_loss_at_epoch = (
                    train_loss_trace[epoch_idx]
                    if epoch_idx < len(train_loss_trace) else None
                )
                internal_loss_records.append({
                    'snapshot': t,
                    'today': t,
                    'tomorrow': t + 1,
                    'epoch': epoch_idx,
                    'val_loss_before_train': epoch_val_loss,
                    'train_loss_after_step': train_loss_at_epoch,
                    'best_val_metric': best_model['val_metric'],
                    'best_val_score': best_model['val_score'],
                    'best_val_loss': best_model['val_loss'],
                    'best_epoch': best_model['train_epoch'],
                    'stopped_epoch': i,
                })
            tqdm.write(
                '[snapshot {}->{}][live update] best_{}={:.4f} '
                'best_val_loss={:.4f} best_epoch={} stopped_epoch={} '
                'unchanged={} last_val_loss={:.4f} last_train_loss={}'.format(
                    t, t + 1, best_model['val_metric'],
                    float(best_model['val_score']), float(best_model['val_loss']),
                    best_model['train_epoch'], i, best_model_unchanged,
                    float(val_loss), train_loss_msg))
            tqdm.write(
                '[snapshot {}->{}][live update] val_loss_trace={} '
                'train_loss_trace={}'.format(
                    t, t + 1,
                    [round(x, 4) for x in val_loss_trace],
                    [round(x, 4) for x in train_loss_trace]))

            # (3) Actually perform the update on training set to get node_states
            # contains information up to time t.
            # Use the best model selected from intra-snapshot cross-validation.
            model.load_state_dict(best_model['state'])

        if cfg.meta.is_meta:  # update meta-learning's initialization weights.
            if model_init is None:  # for the first task.
                model_init = copy.deepcopy(best_model['state'])
            else:  # for subsequent task, update init.
                if cfg.meta.method == 'moving_average':
                    new_weight = cfg.meta.alpha
                elif cfg.meta.method == 'online_mean':
                    new_weight = 1 / (t + 1)  # for t=1, the second item, 1/2.
                else:
                    raise ValueError(f'Invalid method: {cfg.meta.method}')

                # (1-new_weight)*model_init + new_weight*best_model.
                model_init = average_state_dict(model_init,
                                                best_model['state'],
                                                new_weight)

        prev_node_states = update_node_states(model, datasets[0], NET_static[0],
                                              (t, t + 1), prev_node_states,
                                              label_splits[0])
        
    writer.close()

    if cfg.train.ckpt_clean:
        clean_ckpt()

    logging.info('Task done, results saved in {}'.format(cfg.out_dir))

    # # for debugging purpose.
    # import matplotlib.pyplot as plt

    avg_metrics = dict()
    if len(auc_hist) > 0:
        avg_metrics = {
            'LOSS': round(float(np.mean(loss_hist)), 3),
            'AUC': round(float(np.mean(auc_hist)), 3),
            'ACC': round(float(np.mean(acc_hist)), 3),
            'AP': round(float(np.mean(ap_hist)), 3),
        }
        print('Average over all time steps')
        print('LOSS = {:.3f}'.format(avg_metrics['LOSS']))
        print('AUC = {:.3f}'.format(avg_metrics['AUC']))
        print('ACC = {:.3f}'.format(avg_metrics['ACC']))
        print('AP = {:.3f}'.format(avg_metrics['AP']))
    # print('MRR = {:.3f}'.format(np.mean(mrr_hist)))


    '''print("AUC values:", [round(auc, 3) for auc in auc_hist])
    print("MRR values:", [round(mrr, 3) for mrr in mrr_hist])
    print("AP values:", [round(ap, 3) for ap in ap_hist])
    print("ACC values:", [round(acc, 3) for acc in acc_hist])'''

    num_dy = cfg.gnn.layers_mp
    num_st = cfg.gnn.layers_mp1

    data = {
    "LOSS": [round(loss, 3) for loss in loss_hist],
    "AUC": [round(auc, 3) for auc in auc_hist],
    "AP": [round(ap, 3) for ap in ap_hist],
    "ACC": [round(acc, 3) for acc in acc_hist]
    }

    # 创建 DataFrame，index 表示列表的位置 (0, 1, ...)
    df = pd.DataFrame(data)

    yaml_name = os.path.basename(os.path.dirname(cfg.out_dir))
    result_dir = f"/home/ly/SDNL/run/result/{yaml_name}/{cfg.seed}"
    if avg_metrics:
        summary_path = os.path.join(result_dir, 'summary_metrics.json')
        os.makedirs(result_dir, exist_ok=True)
        summary = {
            'yaml_name': yaml_name,
            'seed': cfg.seed,
            'num_dy': num_dy,
            'num_st': num_st,
            'metrics': avg_metrics,
        }
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"平均指标已保存到 {summary_path}")

    for metric in data:
        # 为每个指标生成文件名
        filename = f"dy{num_dy}_st{num_st}_{metric}.xlsx"
        
        # 提取对应指标的数据
        df = pd.DataFrame({metric: data[metric]})
        
        # 设置保存路径
        save_path = os.path.join(result_dir, filename)
        
        # 确保保存路径的目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 保存到 Excel 文件
        df.to_excel(save_path, index=False)

        print(f"{metric} 数据已保存到 {save_path}")

    if internal_loss_records:
        internal_loss_path = os.path.join(
            result_dir, f"dy{num_dy}_st{num_st}_INTERNAL_LOSS.xlsx")
        pd.DataFrame(internal_loss_records).to_excel(
            internal_loss_path, index=False)
        print(f"内部 train/val loss 数据已保存到 {internal_loss_path}")

    df.index.name = "Position"

    # 保存到 Excel 文件
    df.to_excel("output.xlsx", index=True)

    '''print('rck1 = ', np.mean(rck1_hist))
    print('rck3 = ', np.mean(rck3_hist))
    print('rck10 = ', np.mean(rck10_hist))'''
    '''
    df = pd.DataFrame(columns=['时间步', 'AUC', 'MRR'])
    for i in range(len(auc_hist)): 
        df.loc[i] = [i, auc_hist[i], mrr_hist[i]]
    save_path = '/root/autodl-tmp/demo/RT_friend/result/result.xlsx' 
    df.to_excel(save_path, index=False)'''
    '''for i in range(len(auc_hist)):
        print('时间步:',i)
        print('每个时间步的AUC:',auc_hist[i])
        print('每个时间步的MRR:',mrr_hist[i])'''
    # print('最后一天AUC = ', np.mean(auc_hist[-2:]))
    # print('最后一天MRR = ', np.mean(mrr_hist[-2:]))
    # plt.plot(auc_hist, label='AUC')
    # plt.plot(mrr_hist, label='MRR')
    # plt.legend()
    # plt.savefig('temp_out.png')

register_train('live_update', train_live_update)
