"""
A baseline training scheme similar to the training scheme in evolveGCN paper.
"""
import datetime
import logging
import time

import torch
from graphgym.checkpoint import clean_ckpt, load_ckpt, save_ckpt
from graphgym.config import cfg
from graphgym.contrib.train import train_utils
from graphgym.loss import compute_loss
from graphgym.register import register_train
from graphgym.utils.epoch import is_ckpt_epoch, is_eval_epoch
from graphgym.utils.stats import node_degree
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def get_edge_label(dataset, current, horizon, mode):
    if mode == 'before':
        edge_label = torch.cat([dataset[current + i].edge_label
                                for i in range(1, horizon + 1)], dim=0)
        edge_label_index = torch.cat([dataset[current + i].edge_label_index
                                      for i in range(1, horizon + 1)], dim=1)
    elif mode == 'at':
        edge_label = dataset[current + horizon].edge_label
        edge_label_index = dataset[current + horizon].edge_label_index
    return edge_label, edge_label_index


def update_batch(batch, batch_new, mode):

    if mode == 'replace':
        for i in range(len(batch_new.node_states)):
            batch_new.node_states[i] = batch.node_states[i].detach().cpu()
        batch_new.node_degree_existing = batch.node_degree_existing.detach(
        ).cpu()
        return batch_new
    
    elif mode == 'concat':
        keys = ['edge_feature', 'edge_index', 'edge_time']
        for key in keys:
            dim = 1 if 'index' in key else 0
            batch[key] = torch.cat([batch[key], batch_new[key]], dim=dim)
        return batch

class dummy_context_mgr():
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        return False

'''
def train_epoch(logger, model, optimizer, scheduler, dataset, writer, train=True):
    
    with dummy_context_mgr() if train else torch.no_grad():
        if train:
            model.train()
        else:
            model.eval()
        time_start = time.time()

        mrr_lst, rck1_lst, rck3_lst, rck10_lst = [], [], [], []

        optimizer.zero_grad()
        torch.cuda.empty_cache()
        # track cumulative loss over all snapshots.
        accumulated_loss = torch.tensor(0.0).to(torch.device(cfg.device))

        rng = range(len(dataset) - cfg.transaction.horizon)
        for i in tqdm(rng, desc='Snapshot'):
            optimizer.zero_grad()
            torch.cuda.empty_cache()

            if i == 0:
                batch = dataset[i].clone()
            else:
                # update raw features
                batch = update_batch(batch, dataset[i].clone(),
                                     mode='replace')
            batch.node_degree_new = node_degree(
                batch.edge_index,
                n=batch.node_degree_existing.shape[0])

            batch.keep_ratio = train_utils.get_keep_ratio(
                existing=batch.node_degree_existing,
                new=batch.node_degree_new,
                mode=cfg.transaction.keep_ratio)
            batch.keep_ratio = batch.keep_ratio.unsqueeze(-1)
            batch.node_degree_existing += batch.node_degree_new

            # set edge labels
            edge_label, edge_label_index = get_edge_label(dataset, i,
                                                          cfg.transaction.horizon,
                                                          cfg.transaction.pred_mode)

            batch.edge_label = edge_label
            batch.edge_label_index = edge_label_index

            eval_batch = batch.clone()

            batch = train_utils.move_batch_to_device(batch, cfg.device)
            pred, true = model(batch)

            loss, pred_score = compute_loss(pred, true)
            accumulated_loss += loss

            if train and ((i + 1) % cfg.train.tbptt_freq == 0):
                accumulated_loss.backward()
                optimizer.step()

                optimizer.zero_grad()
                torch.cuda.empty_cache()
                # reset the loss tracker.
                accumulated_loss.detach()
                del accumulated_loss
                accumulated_loss = torch.tensor(0.0).to(
                    torch.device(cfg.device))

            # Compute rank based metrics for the current snapshot.
            mrr, rck1, rck3, rck10 = train_utils.report_rank_based_eval(
                eval_batch, model,
                num_neg_per_node=cfg.experimental.rank_eval_multiplier)
            mrr_lst.append(mrr)
            rck1_lst.append(rck1)
            rck3_lst.append(rck3)
            rck10_lst.append(rck10)

            writer.add_scalar('loss', loss, i)
            writer.add_scalar('mrr', mrr, i)
            writer.add_scalar('rck1', rck1, i)
            writer.add_scalar('rck3', rck3, i)
            writer.add_scalar('rck10', rck10, i)

            logger.update_stats(true=true.detach().cpu(),
                                pred=pred_score.detach().cpu(),
                                loss=loss.item(),
                                lr=scheduler.get_last_lr()[0],
                                time_used=time.time() - time_start,
                                params=cfg.params,
                                mrr=mrr, rck1=rck1, rck3=rck3, rck10=rck10)

            time_start = time.time()

        if train and cfg.train.tbptt_freq > 1:
            if (i + 1) % cfg.train.tbptt_freq != 0:
                # use the remaining gradients, useful when num_snapshots does
                # not divide cfg.train.tbptt_freq.
                accumulated_loss.backward()
                optimizer.step()

            scheduler.step()'''

def train_epoch(loggers_list, model, optimizer, scheduler, datasets_list, writer, train=True):
    
    with dummy_context_mgr() if train else torch.no_grad():

        dataset1 = datasets_list[0][0]
        dataset2 = datasets_list[1][0]
        dataset3 = datasets_list[2][0]

        if train:
            model.train()
        else:
            model.eval()
        time_start = time.time()

        mrr_lst, rck1_lst, rck3_lst, rck10_lst = [], [], [], []

        optimizer.zero_grad()
        torch.cuda.empty_cache()
        # track cumulative loss over all snapshots.
        accumulated_loss = torch.tensor(0.0).to(torch.device(cfg.device))

        rng = range(len(dataset1) - cfg.transaction.horizon)

        prev_node_states = []

        for i in tqdm(rng, desc='Snapshot'):
            optimizer.zero_grad()
            torch.cuda.empty_cache()

            if i == 0:
                batch1 = dataset1[i].clone()
                batch2 = dataset2[i].clone()
                batch3 = dataset3[i].clone()
            else:
                # update raw features
                batch1.node_states = prev_node_states[-3]
                batch2.node_states = prev_node_states[-2]
                batch3.node_states = prev_node_states[-1]

                batch1 = update_batch(batch1, dataset1[i].clone(),
                                     mode='replace')
                batch2 = update_batch(batch2, dataset2[i].clone(),
                                     mode='replace')
                batch3 = update_batch(batch3, dataset3[i].clone(),
                                     mode='replace')
                
            batch1.node_degree_new = node_degree(
                batch1.edge_index,
                n=batch1.node_degree_existing.shape[0])
            
            batch2.node_degree_new = node_degree(
                batch2.edge_index,
                n=batch2.node_degree_existing.shape[0])

            batch3.node_degree_new = node_degree(
                batch3.edge_index,
                n=batch3.node_degree_existing.shape[0])

            batch1.keep_ratio = train_utils.get_keep_ratio(
                existing=batch1.node_degree_existing,
                new=batch1.node_degree_new,
                mode=cfg.transaction.keep_ratio)
            
            batch1.keep_ratio = batch1.keep_ratio.unsqueeze(-1)
            batch1.node_degree_existing += batch1.node_degree_new

            batch2.keep_ratio = train_utils.get_keep_ratio(
                existing=batch2.node_degree_existing,
                new=batch2.node_degree_new,
                mode=cfg.transaction.keep_ratio)
            batch2.keep_ratio = batch2.keep_ratio.unsqueeze(-1)
            batch2.node_degree_existing += batch2.node_degree_new

            batch3.keep_ratio = train_utils.get_keep_ratio(
                existing=batch3.node_degree_existing,
                new=batch3.node_degree_new,
                mode=cfg.transaction.keep_ratio)
            batch3.keep_ratio = batch3.keep_ratio.unsqueeze(-1)
            batch3.node_degree_existing += batch3.node_degree_new

            # set edge labels
            edge_label1, edge_label_index1 = get_edge_label(dataset1, i,
                                                          cfg.transaction.horizon,
                                                          cfg.transaction.pred_mode)
            batch1.edge_label = edge_label1
            batch1.edge_label_index = edge_label_index1

            eval_batch1 = batch1.clone()

            batch1 = train_utils.move_batch_to_device(batch1, cfg.device)

            # 对于 batch2
            edge_label2, edge_label_index2 = get_edge_label(dataset2, i,
                                                            cfg.transaction.horizon,
                                                            cfg.transaction.pred_mode)

            batch2.edge_label = edge_label2
            batch2.edge_label_index = edge_label_index2

            eval_batch2 = batch2.clone()

            batch2 = train_utils.move_batch_to_device(batch2, cfg.device)

            # 对于 batch3
            edge_label3, edge_label_index3 = get_edge_label(dataset3, i,
                                                            cfg.transaction.horizon,
                                                            cfg.transaction.pred_mode)

            batch3.edge_label = edge_label3
            batch3.edge_label_index = edge_label_index3

            eval_batch3 = batch3.clone()

            batch3 = train_utils.move_batch_to_device(batch3, cfg.device)

            eval_batchs = [eval_batch1, eval_batch2, eval_batch3]

            (pred1, true1), (pred2, true2), (pred3, true3), (states1, states2, states3) = model(batch1, batch2, batch3)

            prev_node_states.append(states1)
            prev_node_states.append(states2)
            prev_node_states.append(states3)

            loss1, pred_score1 = compute_loss(pred1, true1)
            loss2, pred_score2 = compute_loss(pred2, true2)
            loss3, pred_score3 = compute_loss(pred3, true3)

            loss = loss1 + loss2 + loss3 

            accumulated_loss += loss

            if train and ((i + 1) % cfg.train.tbptt_freq == 0):
                accumulated_loss.backward()
                optimizer.step()

                optimizer.zero_grad()
                torch.cuda.empty_cache()
                # reset the loss tracker.
                accumulated_loss.detach()
                del accumulated_loss
                accumulated_loss = torch.tensor(0.0).to(
                    torch.device(cfg.device))

            # Compute rank based metrics for the current snapshot.
            mrr, rck1, rck3, rck10 = train_utils.report_rank_based_eval1(
                eval_batchs, model,
                num_neg_per_node=cfg.experimental.rank_eval_multiplier)

            mrr_lst.append(mrr)
            rck1_lst.append(rck1)
            rck3_lst.append(rck3)
            rck10_lst.append(rck10)

            writer.add_scalar('loss', loss, i)
            writer.add_scalar('mrr', mrr, i)
            writer.add_scalar('rck1', rck1, i)
            writer.add_scalar('rck3', rck3, i)
            writer.add_scalar('rck10', rck10, i)

            logger1 = loggers_list[0][0]
            logger2 = loggers_list[1][0]
            logger3 = loggers_list[2][0]           

            logger1.update_stats(true=true1.detach().cpu(),
                                pred=pred_score1.detach().cpu(),
                                loss=loss1.item(),
                                lr=scheduler.get_last_lr()[0],
                                time_used=time.time() - time_start,
                                params=cfg.params,
                                mrr=mrr, rck1=rck1, rck3=rck3, rck10=rck10)
            
            logger2.update_stats(true=true2.detach().cpu(),
                                pred=pred_score2.detach().cpu(),
                                loss=loss2.item(),
                                lr=scheduler.get_last_lr()[0],
                                time_used=time.time() - time_start,
                                params=cfg.params,
                                mrr=mrr, rck1=rck1, rck3=rck3, rck10=rck10)
            
            logger3.update_stats(true=true3.detach().cpu(),
                                pred=pred_score3.detach().cpu(),
                                loss=loss3.item(),
                                lr=scheduler.get_last_lr()[0],
                                time_used=time.time() - time_start,
                                params=cfg.params,
                                mrr=mrr, rck1=rck1, rck3=rck3, rck10=rck10)

            time_start = time.time()

        if train and cfg.train.tbptt_freq > 1:
            if (i + 1) % cfg.train.tbptt_freq != 0:
                # use the remaining gradients, useful when num_snapshots does
                # not divide cfg.train.tbptt_freq.
                accumulated_loss.backward()
                optimizer.step()

            scheduler.step()

def train_example(loggers_list, loaders_list, model, optimizer, scheduler, datasets_list,
                  **kwargs):
    start_epoch = 0
    if cfg.train.auto_resume:
        start_epoch = load_ckpt(model, optimizer, scheduler)
    if start_epoch == cfg.optim.max_epoch:
        logging.info('Checkpoint found, Task already done')
    else:
        logging.info('Start from epoch {}'.format(start_epoch))

    t = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
    # directory to store tensorboard files of this run.
    out_dir = t + '_' + cfg.out_dir.replace('/', '\\')
    # dir to store all run outputs for the entire batch.
    run_dir = 'runs_' + cfg.remark

    print(f'Tensorboard directory: {out_dir}')

    num_splits1 = len(loggers_list[0])
    num_splits2 = len(loggers_list[1])
    num_splits3 = len(loggers_list[2])

    for cur_epoch in range(start_epoch, cfg.optim.max_epoch):
        train_writer = SummaryWriter(f'./{run_dir}/{out_dir}/train_{cur_epoch}')
        val_writer = SummaryWriter(f'./{run_dir}/{out_dir}/val_{cur_epoch}')
        test_writer = SummaryWriter(f'./{run_dir}/{out_dir}/test_{cur_epoch}')

        '''train_epoch(loggers[0], model, optimizer, scheduler,
                    datasets[0], train_writer, train=True)'''
        
        train_epoch(loggers_list, model, optimizer, scheduler,
                    datasets_list, train_writer, train=True)
        
        loggers[0].write_epoch(cur_epoch)
        if is_eval_epoch(cur_epoch):
            for i, writer in zip(range(1, num_splits), [val_writer, test_writer]):
                train_epoch(loggers[i], model, optimizer, scheduler,
                            datasets[i], writer, train=False)
                loggers[i].write_epoch(cur_epoch)
        if is_ckpt_epoch(cur_epoch):
            save_ckpt(model, optimizer, scheduler, cur_epoch)

    for logger in loggers:
        logger.close()
    if cfg.train.ckpt_clean:
        clean_ckpt()

    logging.info('Task done, results saved in {}'.format(cfg.out_dir))

'''
def train_example(logger, loader, model, optimizer, scheduler, dataset,
                  **kwargs):
    start_epoch = 0
    if cfg.train.auto_resume:
        start_epoch = load_ckpt(model, optimizer, scheduler)
    if start_epoch == cfg.optim.max_epoch:
        logging.info('Checkpoint found, Task already done')
    else:
        logging.info('Start from epoch {}'.format(start_epoch))

    t = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
    # directory to store tensorboard files of this run.
    out_dir = t + '_' + cfg.out_dir.replace('/', '\\')
    # dir to store all run outputs for the entire batch.
    run_dir = 'runs_' + cfg.remark

    print(f'Tensorboard directory: {out_dir}')

    num_splits = len(loggers)
    for cur_epoch in range(start_epoch, cfg.optim.max_epoch):
        train_writer = SummaryWriter(f'./{run_dir}/{out_dir}/train_{cur_epoch}')
        val_writer = SummaryWriter(f'./{run_dir}/{out_dir}/val_{cur_epoch}')
        test_writer = SummaryWriter(f'./{run_dir}/{out_dir}/test_{cur_epoch}')

        #train_epoch(loggers[0], model, optimizer, scheduler,
                    # datasets[0], train_writer, train=True)
        train_epoch(loggers_list, model, optimizer, scheduler,
                    datasets_list, train_writer, train=True)
        
        loggers[0].write_epoch(cur_epoch)
        if is_eval_epoch(cur_epoch):
            for i, writer in zip(range(1, num_splits), [val_writer, test_writer]):
                train_epoch(loggers[i], model, optimizer, scheduler,
                            datasets[i], writer, train=False)
                loggers[i].write_epoch(cur_epoch)
        if is_ckpt_epoch(cur_epoch):
            save_ckpt(model, optimizer, scheduler, cur_epoch)

    for logger in loggers:
        logger.close()
    if cfg.train.ckpt_clean:
        clean_ckpt()

    logging.info('Task done, results saved in {}'.format(cfg.out_dir))'''

register_train('baseline', train_example)