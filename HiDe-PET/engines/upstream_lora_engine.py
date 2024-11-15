import math
import os
from pathlib import Path
import sys
from typing import Iterable
import numpy as np
import torch
from timm.optim import create_optimizer
import utils
from timm.utils import accuracy
import torch.distributed as dist
from torch.distributions.multivariate_normal import MultivariateNormal
from torch import optim
from timm.scheduler import create_scheduler

def train_and_evaluate_continual_model(continual_model, model_without_ddp, vanilla_model, criterion, data_loader, optimizer, lr_scheduler, device, class_mask, target_dataset_map, target_task_map, task_dataset_map, args):
    acc_matrix = np.zeros((args.num_tasks, args.num_tasks))

    for task_id in range(args.num_tasks):
        if task_id > 0 and args.reinit_optimizer:
            base_params = [p for name, p in model_without_ddp.named_parameters() if 'lora' in name and p.requires_grad == True]
            base_fc_params = [p for name, p in model_without_ddp.named_parameters() if 'lora' not in name and p.requires_grad == True]
            base_params = {'params': base_params, 'lr': args.lr * 0.1, 'weight_decay': args.weight_decay}
            base_fc_params = {'params': base_fc_params, 'lr': args.lr, 'weight_decay': args.weight_decay}
            network_params = [base_params, base_fc_params]
            optimizer = create_optimizer(args, network_params)
        
        for epoch in range(args.epochs): 
            train_stats = train_one_epoch(model=continual_model, criterion=criterion, data_loader=data_loader[task_id]['train'], optimizer=optimizer, 
                                          device=device, epoch=epoch, task_dataset_map=task_dataset_map, max_norm=args.clip_grad, set_training_mode=True, task_id=task_id, 
                                          class_mask=class_mask, args=args, training_type='vanilla')
    
            if lr_scheduler:
                lr_scheduler.step(epoch)
        
        if args.continual_model_output_dir and utils.is_main_process():
            Path(os.path.join(args.continual_model_output_dir, 'checkpoint')).mkdir(parents=True, exist_ok=True)

            checkpoint_path = os.path.join(args.continual_model_output_dir, 'checkpoint/task{}_checkpoint.pth'.format(task_id + 1))
            state_dict = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'args': args,
            }
            if args.sched is not None and args.sched != 'constant':
                state_dict['lr_scheduler'] = lr_scheduler.state_dict()

            utils.save_on_master(state_dict, checkpoint_path)

        test_stats = evaluate_till_now(model=continual_model, data_loader=data_loader,
                                       device=device, task_id=task_id, class_mask=class_mask,
                                       acc_matrix=acc_matrix, args=args, 
                                       evaluate_type='vanilla', original_model=vanilla_model,
                                       target_dataset_map=target_dataset_map, 
                                       target_task_map=target_task_map, task_dataset_map=task_dataset_map)


def train_and_evaluate_vanilla_model(vanilla_model, model_without_ddp, criterion, data_loader, data_loader_per_cls, optimizer, lr_scheduler, device, class_mask, target_dataset_map, target_task_map, task_dataset_map, args):
    acc_matrix = np.zeros((args.num_tasks, args.num_tasks))
    pre_ca_acc_matrix = np.zeros((args.num_tasks, args.num_tasks))
    global cls_mean
    global cls_cov
    cls_mean = dict()
    cls_cov = dict()

    for task_id in range(args.num_tasks):
        if task_id > 0 and args.reinit_optimizer:
            args.lr = args.cl_lr[task_dataset_map[task_id]]
            optimizer = create_optimizer(args, model_without_ddp)
        
        for epoch in range(args.epochs): 
            train_stats = train_one_epoch(model=vanilla_model, criterion=criterion, data_loader=data_loader[task_id]['train'], optimizer=optimizer, 
                                          device=device, epoch=epoch, task_dataset_map=task_dataset_map, max_norm=args.clip_grad, set_training_mode=True, task_id=task_id, 
                                          class_mask=class_mask, args = args, training_type='vanilla')
    
            if lr_scheduler:
                lr_scheduler.step(epoch)
        _compute_mean(model=vanilla_model, data_loader=data_loader_per_cls, device=device, task_id=task_id,
                      class_mask=class_mask[task_id], args=args)
        
        if task_id > 0:
            pre_ca_test_stats = evaluate_till_now(model=vanilla_model, data_loader=data_loader,
                                                  device=device, task_id=task_id, class_mask=class_mask,
                                                  acc_matrix=acc_matrix, args=args, 
                                                  evaluate_type='vanilla', original_model=vanilla_model,
                                                  target_dataset_map=target_dataset_map, 
                                                  target_task_map=target_task_map, task_dataset_map=task_dataset_map)
            train_task_adaptive_prediction(vanilla_model, args, device, class_mask, task_id)

        test_stats = evaluate_till_now(model=vanilla_model, data_loader=data_loader,
                                       device=device, task_id=task_id, class_mask=class_mask,
                                       acc_matrix=acc_matrix, args=args, 
                                       evaluate_type='vanilla', original_model=vanilla_model,
                                       target_dataset_map=target_dataset_map, 
                                       target_task_map=target_task_map, task_dataset_map=task_dataset_map)
        
        if args.vanilla_model_output_dir and utils.is_main_process():
            Path(os.path.join(args.vanilla_model_output_dir, 'checkpoint')).mkdir(parents=True, exist_ok=True)

            checkpoint_path = os.path.join(args.vanilla_model_output_dir, 'checkpoint/task{}_checkpoint.pth'.format(task_id + 1))
            state_dict = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'args': args,
            }
            if args.sched is not None and args.sched != 'constant':
                state_dict['lr_scheduler'] = lr_scheduler.state_dict()

            utils.save_on_master(state_dict, checkpoint_path)


def train_and_evaluate_shared_model(shared_model, model_without_ddp, vanilla_model, criterion, data_loader, data_loader_per_cls, optimizer, lr_scheduler, device, class_mask, target_dataset_map, target_task_map, task_dataset_map, args):
    acc_matrix = np.zeros((args.num_tasks, args.num_tasks))
    pre_ca_acc_matrix = np.zeros((args.num_tasks, args.num_tasks))
    global cls_mean
    global cls_cov
    cls_mean = dict()
    cls_cov = dict()

    for task_id in range(args.num_tasks):
        if task_id > 0 and args.reinit_optimizer:      
            base_params = [p for name, p in model_without_ddp.named_parameters() if 'lora' in name and p.requires_grad == True]
            base_fc_params = [p for name, p in model_without_ddp.named_parameters() if 'lora' not in name and p.requires_grad == True]
            if task_id % 5 == 0:
                base_params = {'params': base_params, 'lr': args.cl_lr[task_dataset_map[task_id]], 'weight_decay': args.weight_decay}
            else:
                base_params = {'params': base_params, 'lr': args.cl_lr[task_dataset_map[task_id]] * 0.1, 'weight_decay': args.weight_decay}

            base_fc_params = {'params': base_fc_params, 'lr': args.lr, 'weight_decay': args.weight_decay}
            network_params = [base_params, base_fc_params]
            optimizer = create_optimizer(args, network_params)
            if args.sched != 'constant':
                lr_scheduler, _ = create_scheduler(args, optimizer)
            elif args.sched == 'constant':
                lr_scheduler = None
        
        original_checkpoint_path = os.path.join(args.vanilla_model_output_dir, 'checkpoint/task{}_checkpoint.pth'.format(task_id + 1))
        if os.path.exists(original_checkpoint_path):
            print('Loading checkpoint from:', original_checkpoint_path)
            original_checkpoint = torch.load(original_checkpoint_path, map_location=device)
            vanilla_model.load_state_dict(original_checkpoint['model'], strict=True)
        else:
            print('No checkpoint found at:', original_checkpoint_path)
            return
        
        if task_id > 0 and task_id % 5 == 0:
            j = task_dataset_map[task_id]
            with torch.no_grad():
                if args.distributed:
                    shared_model.module.lora_layer.k_lora_A.grad.zero_()
                    shared_model.module.lora_layer.k_lora_A[j] = shared_model.module.lora_layer.k_lora_A[j-1]
                    shared_model.module.lora_layer.k_lora_B.grad.zero_()
                    shared_model.module.lora_layer.k_lora_B[j] = shared_model.module.lora_layer.k_lora_B[j-1]
                    shared_model.module.lora_layer.v_lora_A.grad.zero_()
                    shared_model.module.lora_layer.v_lora_A[j] = shared_model.module.lora_layer.v_lora_A[j-1]
                    shared_model.module.lora_layer.v_lora_B.grad.zero_()
                    shared_model.module.lora_layer.v_lora_B[j] = shared_model.module.lora_layer.v_lora_B[j-1]
                else:
                    shared_model.lora_layer.k_lora_A.grad.zero_()
                    shared_model.lora_layer.k_lora_A[j] = shared_model.lora_layer.k_lora_A[j-1]
                    shared_model.lora_layer.k_lora_B.grad.zero_()
                    shared_model.lora_layer.k_lora_B[j] = shared_model.lora_layer.k_lora_B[j-1]
                    shared_model.lora_layer.v_lora_A.grad.zero_()
                    shared_model.lora_layer.v_lora_A[j] = shared_model.lora_layer.v_lora_A[j-1]
                    shared_model.lora_layer.v_lora_B.grad.zero_()
                    shared_model.lora_layer.v_lora_B[j] = shared_model.lora_layer.v_lora_B[j-1]
        
        for epoch in range(args.epochs): 
            train_stats = train_one_epoch(model=shared_model, criterion=criterion, data_loader=data_loader[task_id]['train'], optimizer=optimizer, 
                                          device=device, epoch=epoch, task_dataset_map=task_dataset_map, max_norm=args.clip_grad, set_training_mode=True, task_id=task_id, 
                                          class_mask=class_mask, args=args, training_type='shared')
    
            if lr_scheduler:
                lr_scheduler.step(epoch)
        _compute_mean(model=shared_model, data_loader=data_loader_per_cls, device=device, task_id=task_dataset_map[task_id],
                      class_mask=class_mask[task_id], args=args)


        if task_id > 0:
            pre_ca_test_stats = evaluate_till_now(model=shared_model, data_loader=data_loader,
                                                  device=device, task_id=task_id, class_mask=class_mask,
                                                  acc_matrix=acc_matrix, args=args, 
                                                  evaluate_type='shared', original_model=vanilla_model,
                                                  target_dataset_map=target_dataset_map, 
                                                  target_task_map=target_task_map, task_dataset_map=task_dataset_map)
            train_task_adaptive_prediction(shared_model, args, device, class_mask, task_id)

        test_stats = evaluate_till_now(model=shared_model, data_loader=data_loader,
                                       device=device, task_id=task_id, class_mask=class_mask,
                                       acc_matrix=acc_matrix, args=args, 
                                       evaluate_type='shared', original_model=vanilla_model,
                                       target_dataset_map=target_dataset_map, 
                                       target_task_map=target_task_map, task_dataset_map=task_dataset_map)
        
        if args.shared_model_output_dir and utils.is_main_process():
            Path(os.path.join(args.shared_model_output_dir, 'checkpoint')).mkdir(parents=True, exist_ok=True)

            checkpoint_path = os.path.join(args.shared_model_output_dir, 'checkpoint/task{}_checkpoint.pth'.format(task_id + 1))
            state_dict = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'args': args,
            }
            if args.sched is not None and args.sched != 'constant':
                state_dict['lr_scheduler'] = lr_scheduler.state_dict()

            utils.save_on_master(state_dict, checkpoint_path)


def train_one_epoch(model: torch.nn.Module, criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, task_dataset_map: dict, max_norm: float = 0,
                    set_training_mode=True, task_id=-1, class_mask=None, args=None, training_type='vanilla'):
    model.train(set_training_mode)
    
    if args.distributed and utils.get_world_size() > 1:
        data_loader.sampler.set_epoch(epoch)
        
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = f'Train: Epoch[{epoch + 1:{int(math.log10(args.epochs)) + 1}}/{args.epochs}]'

    for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        if training_type == 'shared':
            output = model(input, task_id=task_dataset_map[task_id], train=set_training_mode)
        else:
            output = model(input, task_id=task_id, train=set_training_mode)
        logits = output['logits']

        # here is the trick to mask out classes of non-current tasks
        if args.train_mask and class_mask is not None:
            mask = class_mask[task_id]
            not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
            not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
            logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))

        loss = criterion(logits, target)  # base criterion (CrossEntropyLoss)

        acc1, acc5 = accuracy(logits, target, topk=(1, 5))

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        torch.cuda.synchronize()
        metric_logger.update(Loss=loss.item())
        metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
        metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model: torch.nn.Module, original_model: torch.nn.Module, data_loader,
             device, i=-1, task_id=-1, class_mask=None, target_task_map=None, args=None, evaluate_type='vanilla',
             target_dataset_map=None, task_dataset_map=None, mixture_model=None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test: [Task {}]'.format(i + 1)

    # switch to evaluation mode
    model.eval()
    original_model.eval()

    with torch.no_grad():
        for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            if evaluate_type == 'vanilla':
                output = model(input)
                logits = output['logits']
                if args.train_mask and class_mask is not None:
                    mask = []
                    for id in range(task_id + 1):
                        mask.extend(class_mask[id])
                    not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
                    not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
                    logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))
                lora_id = torch.max(logits, dim=1)[1] 
                lora_id = torch.tensor([target_task_map[v.item()] for v in lora_id], device=device)
                task_inference_acc = utils.task_inference_accuracy(lora_id.unsqueeze(-1), target, target_task_map)
            
            elif evaluate_type == 'shared':
                with torch.no_grad():
                    if original_model is not None:
                        output = original_model(input)
                        logits = output['logits']
                        if args.train_mask and class_mask is not None:
                            mask = []
                            for id in range(task_id + 1):
                                mask.extend(class_mask[id])
                            not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
                            not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
                            logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))
                        lora_id = torch.max(logits, dim=1)[1]
                        lora_id = torch.tensor([target_dataset_map[v.item()] for v in lora_id], device=device)
                        task_inference_acc = utils.task_inference_accuracy(lora_id.unsqueeze(-1), target, target_dataset_map)
                    else:
                        raise NotImplementedError("original model is None")
                    output = model(input, task_id=lora_id)
                    logits = output['logits']

            elif evaluate_type == 'hide':
                with torch.no_grad():
                    if original_model is not None:
                        output = original_model(input)
                        logits = output['logits']
                        if args.train_mask and class_mask is not None:
                            mask = []
                            for id in range(task_id + 1):
                                mask.extend(class_mask[id])
                            not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
                            not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
                            logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))
                        lora_id = torch.max(logits, dim=1)[1]
                        lora_id = torch.tensor([target_task_map[v.item()] for v in lora_id], device=device)
                        # print(lora_id)
                        task_inference_acc = utils.task_inference_accuracy(lora_id.unsqueeze(-1), target, target_task_map)
                    else:
                        raise NotImplementedError("original model is None")

                    output = model(input, task_id=lora_id)
                    logits = output['logits']

            else:
                with torch.no_grad():
                    if original_model is not None:
                        output = original_model(input)
                        logits = output['logits']
                        if args.train_mask and class_mask is not None:
                            mask = []
                            for id in range(task_id + 1):
                                mask.extend(class_mask[id])
                            not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
                            not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
                            logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))
                        lora_id = torch.max(logits, dim=1)[1]
                        lora_shared_id = torch.tensor([target_dataset_map[v.item()] for v in lora_id], device=device)
                        lora_mixture_id = torch.tensor([target_task_map[v.item()] for v in lora_id], device=device)
                        # print(lora_id)
                        task_inference_acc = utils.task_inference_accuracy(lora_mixture_id.unsqueeze(-1), target, target_task_map)
                    else:
                        raise NotImplementedError("original model is None")
                    # shared_model update attention
                    #lora_shared_id = torch.argmax(torch.bincount(lora_shared_id))
                    #if lora_shared_id > 0:
                    #    lora_shared_id -= 1
                    #model.update_attention(task_id=lora_shared_id, device=device)
                    #_copy_backbone_parameters(mixture_model.module, model, args)
                    #_copy_backbone_parameters(model, original_model, args)
                    output = mixture_model(input, task_id=lora_mixture_id)
                    logits = output['logits']


            if args.task_inc and class_mask is not None:
                # adding mask to output logits
                mask = class_mask[i]
                mask = torch.tensor(mask, dtype=torch.int64).to(device)
                logits_mask = torch.ones_like(logits, device=device) * float('-inf')
                logits_mask = logits_mask.index_fill(1, mask, 0.0)
                logits = logits + logits_mask

            loss = criterion(logits, target)

            acc1, acc5 = accuracy(logits, target, topk=(1, 5))

            metric_logger.meters['Loss'].update(loss.item())
            metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
            metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])
            metric_logger.meters['Acc@task'].update(task_inference_acc.item(), n=input.shape[0])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(
        '* Acc@task {task_acc.global_avg:.3f} Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
        .format(task_acc=metric_logger.meters['Acc@task'],
                top1=metric_logger.meters['Acc@1'], top5=metric_logger.meters['Acc@5'],
                losses=metric_logger.meters['Loss']))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_till_now(model: torch.nn.Module, original_model: torch.nn.Module, data_loader,
                      device, task_id=-1, class_mask=None, target_task_map=None, 
                      target_dataset_map=None, task_dataset_map=None,
                      acc_matrix=None, args=None, evaluate_type='vanilla', mixture_model=None):
    stat_matrix = np.zeros((4, args.num_tasks))  # 3 for Acc@1, Acc@5, Loss

    for i in range(task_id + 1):
        test_stats = evaluate(model=model, original_model=original_model, data_loader=data_loader[i]['val'],
                              device=device, i=i, task_id=task_id, class_mask=class_mask, target_task_map=target_task_map,
                              args=args, evaluate_type=evaluate_type, target_dataset_map=target_dataset_map, task_dataset_map=task_dataset_map, mixture_model=mixture_model)

        stat_matrix[0, i] = test_stats['Acc@1']
        stat_matrix[1, i] = test_stats['Acc@5']
        stat_matrix[2, i] = test_stats['Loss']
        stat_matrix[3, i] = test_stats['Acc@task']

        acc_matrix[i, task_id] = test_stats['Acc@1']

    avg_stat = np.divide(np.sum(stat_matrix, axis=1), task_id + 1)

    diagonal = np.diag(acc_matrix)

    result_str = "[Average accuracy till task{}]\tAcc@task: {:.4f}\tAcc@1: {:.4f}\tAcc@5: {:.4f}\tLoss: {:.4f}".format(
        task_id + 1,
        avg_stat[3],
        avg_stat[0],
        avg_stat[1],
        avg_stat[2])
    if task_id > 0:
        forgetting = np.mean((np.max(acc_matrix, axis=1) -
                              acc_matrix[:, task_id])[:task_id])
        backward = np.mean((acc_matrix[:, task_id] - diagonal)[:task_id])

        result_str += "\tForgetting: {:.4f}\tBackward: {:.4f}".format(forgetting, backward)
    print(result_str)

    return test_stats


@torch.no_grad()
def _compute_mean(model: torch.nn.Module, data_loader: Iterable, device: torch.device, task_id, class_mask=None,
                  args=None, ):
    model.eval()

    for cls_id in class_mask:
        data_loader_cls = data_loader[cls_id]['train']
        features_per_cls = []
        for i, (inputs, targets) in enumerate(data_loader_cls):
            inputs = inputs.to(device, non_blocking=True)
            features = model(inputs, task_id=task_id, train=True)['pre_logits']
            features_per_cls.append(features)
        features_per_cls = torch.cat(features_per_cls, dim=0)
        features_per_cls_list = [torch.zeros_like(features_per_cls, device=device) for _ in range(args.world_size)]

        dist.barrier()
        dist.all_gather(features_per_cls_list, features_per_cls)

        if args.ca_storage_efficient_method == 'covariance':
            features_per_cls = torch.cat(features_per_cls_list, dim=0)
            # print(features_per_cls.shape)
            cls_mean[cls_id] = features_per_cls.mean(dim=0)
            cls_cov[cls_id] = torch.cov(features_per_cls.T) + (torch.eye(cls_mean[cls_id].shape[-1]) * 1e-4).to(device)
        
        if args.ca_storage_efficient_method == 'variance':
            features_per_cls = torch.cat(features_per_cls_list, dim=0)
            # print(features_per_cls.shape)
            cls_mean[cls_id] = features_per_cls.mean(dim=0)
            cls_cov[cls_id] = torch.diag(torch.cov(features_per_cls.T) + (torch.eye(cls_mean[cls_id].shape[-1]) * 1e-4).to(device))
        if args.ca_storage_efficient_method == 'multi-centroid':
            from sklearn.cluster import KMeans
            n_clusters = args.n_centroids
            features_per_cls = torch.cat(features_per_cls_list, dim=0).cpu().numpy()
            kmeans = KMeans(n_clusters=n_clusters)
            kmeans.fit(features_per_cls)
            cluster_lables = kmeans.labels_
            cluster_means = []
            cluster_vars = []
            actual_cluster = n_clusters
            for i in range(n_clusters):
               cluster_data = features_per_cls[cluster_lables == i]
               cluster_mean = torch.tensor(np.mean(cluster_data, axis=0), dtype=torch.float64).to(device)
               cluster_var = torch.tensor(np.var(cluster_data, axis=0), dtype=torch.float64).to(device)
               cluster_means.append(cluster_mean)
               cluster_vars.append(cluster_var)
               if cluster_var.mean() == 0:
                    actual_cluster -= 1
            
            cls_mean[cls_id] = cluster_means
            cls_cov[cls_id] = cluster_vars
            

        if args.ca_storage_efficient_method == 'prototype':
            # cls_mean[cls_id] save args.n_centroids prototypes
            import random
            features_per_cls = torch.cat(features_per_cls_list, dim=0)
            print(features_per_cls.shape)
            idx = random.sample(range(features_per_cls.shape[0]), args.n_centroids)
            cls_mean[cls_id] = [features_per_cls[i] for i in idx]



def train_task_adaptive_prediction(model: torch.nn.Module, args, device, class_mask=None, task_id=-1):
    model.train()
    run_epochs = args.crct_epochs
    crct_num = 0
    param_list = [p for n, p in model.named_parameters() if p.requires_grad and 'prompt' not in n]
    network_params = [{'params': param_list, 'lr': args.ca_lr, 'weight_decay': args.weight_decay}]
    if 'mae' in args.model or 'beit' in args.model:
        optimizer = optim.AdamW(network_params, lr=args.ca_lr / 10, weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(network_params, lr=args.ca_lr, momentum=0.9, weight_decay=5e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=run_epochs)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    for i in range(task_id):
        crct_num += len(class_mask[i])

    # TODO: efficiency may be improved by encapsulating sampled data into Datasets class and using distributed sampler.
    for epoch in range(run_epochs):

        sampled_data = []
        sampled_label = []
        num_sampled_pcls = args.batch_size * 5

        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))

        if args.ca_storage_efficient_method in ['covariance', 'variance']:
            for i in range(task_id + 1):
                for c_id in class_mask[i]:
                    mean = torch.tensor(cls_mean[c_id], dtype=torch.float64).to(device)
                    cov = cls_cov[c_id].to(device)
                    if args.ca_storage_efficient_method == 'variance':
                        cov = torch.diag(cov)
                    m = MultivariateNormal(mean.float(), cov.float())
                    sampled_data_single = m.sample(sample_shape=(num_sampled_pcls,))
                    sampled_data.append(sampled_data_single)

                    sampled_label.extend([c_id] * num_sampled_pcls)

        elif args.ca_storage_efficient_method == 'multi-centroid':
            for i in range(task_id + 1):
               for c_id in class_mask[i]:
                   for cluster in range(len(cls_mean[c_id])):
                       mean = cls_mean[c_id][cluster]
                       var = cls_cov[c_id][cluster]
                       if var.mean() == 0:
                           continue
                       m = MultivariateNormal(mean.float(), (torch.diag(var) + 1e-4 * torch.eye(mean.shape[0]).to(mean.device)).float())
                       sampled_data_single = m.sample(sample_shape=(num_sampled_pcls,))
                       sampled_data.append(sampled_data_single)
                       sampled_label.extend([c_id] * num_sampled_pcls)

        elif args.ca_storage_efficient_method == 'prototype':
            for i in range(task_id + 1):
                for c_id in class_mask[i]:
                    for proto_idx in range(len(cls_mean[c_id])):
                        prototype = cls_mean[c_id][proto_idx]
                        m = MultivariateNormal(torch.tensor([0]*prototype.shape[0], device=prototype.device).float(), torch.eye(prototype.shape[0]).to(prototype.device).float())
                        sampled_data_single = m.sample(sample_shape=(num_sampled_pcls,)) + prototype
                        sampled_data.append(sampled_data_single)
                        sampled_label.extend([c_id] * num_sampled_pcls)
        else:
            raise NotImplementedError


        sampled_data = torch.cat(sampled_data, dim=0).float().to(device)
        sampled_label = torch.tensor(sampled_label).long().to(device)
        print(sampled_data.shape)

        inputs = sampled_data
        targets = sampled_label

        sf_indexes = torch.randperm(inputs.size(0))
        inputs = inputs[sf_indexes]
        targets = targets[sf_indexes]
        # print(targets)

        for _iter in range(crct_num):
            inp = inputs[_iter * num_sampled_pcls:(_iter + 1) * num_sampled_pcls]
            tgt = targets[_iter * num_sampled_pcls:(_iter + 1) * num_sampled_pcls]
            outputs = model(inp, fc_only=True)
            logits = outputs['logits']

            if args.train_mask and class_mask is not None:
                mask = []
                for id in range(task_id + 1):
                    mask.extend(class_mask[id])
                # print(mask)
                not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
                not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
                logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))

            loss = criterion(logits, tgt)  # base criterion (CrossEntropyLoss)
            acc1, acc5 = accuracy(logits, tgt, topk=(1, 5))

            if not math.isfinite(loss.item()):
                print("Loss is {}, stopping training".format(loss.item()))
                sys.exit(1)

            optimizer.zero_grad()
            loss.backward()
            #for name, p in model.named_parameters():
            #    if p.requires_grad and p.grad is None:
            #        print(name)
            optimizer.step()
            torch.cuda.synchronize()

            metric_logger.update(Loss=loss.item())
            metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
            metric_logger.meters['Acc@1'].update(acc1.item(), n=inp.shape[0])
            metric_logger.meters['Acc@5'].update(acc5.item(), n=inp.shape[0])

            # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        scheduler.step()


def train_and_evaluate_hide_model(model, model_without_ddp, vanilla_model, criterion, data_loader, data_loader_per_cls, optimizer, lr_scheduler, device, class_mask, target_dataset_map, target_task_map, task_dataset_map, args):
    acc_matrix = np.zeros((args.num_tasks, args.num_tasks))
    pre_ca_acc_matrix = np.zeros((args.num_tasks, args.num_tasks))
    global cls_mean
    global cls_cov
    cls_mean = dict()
    cls_cov = dict()

    for task_id in range(args.num_tasks):
        if task_id > 0 and args.reinit_optimizer:
            base_params = [p for name, p in model_without_ddp.named_parameters() if 'lora' in name and p.requires_grad == True]
            base_fc_params = [p for name, p in model_without_ddp.named_parameters() if 'lora' not in name and p.requires_grad == True]
            base_params = {'params': base_params, 'lr': args.lr, 'weight_decay': args.weight_decay}
            base_fc_params = {'params': base_fc_params, 'lr': args.lr, 'weight_decay': args.weight_decay}
            network_params = [base_params, base_fc_params]
            optimizer = create_optimizer(args, network_params)
            if args.sched != 'constant':
                lr_scheduler, _ = create_scheduler(args, optimizer)
            elif args.sched == 'constant':
                lr_scheduler = None
        
        original_checkpoint_path = os.path.join(args.vanilla_model_output_dir, 'checkpoint/task{}_checkpoint.pth'.format(task_id + 1))
        if os.path.exists(original_checkpoint_path):
            print('Loading checkpoint from:', original_checkpoint_path)
            original_checkpoint = torch.load(original_checkpoint_path, map_location=device)
            vanilla_model.load_state_dict(original_checkpoint['model'], strict=True)
        else:
            print('No checkpoint found at:', original_checkpoint_path)
            return
        if args.eval:
            hide_checkpoint_path = os.path.join(args.output_dir, 'checkpoint/task{}_checkpoint.pth'.format(task_id + 1))
            if os.path.exists(hide_checkpoint_path):
                print('Loading checkpoint from:', hide_checkpoint_path)
                hide_checkpoint = torch.load(hide_checkpoint_path, map_location=device)
                model_without_ddp.load_state_dict(hide_checkpoint['model'], strict=True)
        else:
            if task_id > 0:
                with torch.no_grad():
                    if args.distributed:
                        model.module.lora_layer.k_lora_A.grad.zero_()
                        model.module.lora_layer.k_lora_A[task_id] = model.module.lora_layer.k_lora_A[task_id-1]
                        model.module.lora_layer.k_lora_B.grad.zero_()
                        model.module.lora_layer.k_lora_B[task_id] = model.module.lora_layer.k_lora_B[task_id-1]
                        model.module.lora_layer.v_lora_A.grad.zero_()
                        model.module.lora_layer.v_lora_A[task_id] = model.module.lora_layer.v_lora_A[task_id-1]
                        model.module.lora_layer.v_lora_B.grad.zero_()
                        model.module.lora_layer.v_lora_B[task_id] = model.module.lora_layer.v_lora_B[task_id-1]
                    else:
                        model.lora_layer.k_lora_A.grad.zero_()
                        model.lora_layer.k_lora_A[task_id] = model.lora_layer.k_lora_A[task_id-1]
                        model.lora_layer.k_lora_B.grad.zero_()
                        model.lora_layer.k_lora_B[task_id] = model.lora_layer.k_lora_B[task_id-1]
                        model.lora_layer.v_lora_A.grad.zero_()
                        model.lora_layer.v_lora_A[task_id] = model.lora_layer.v_lora_A[task_id-1]
                        model.lora_layer.v_lora_B.grad.zero_()
                        model.lora_layer.v_lora_B[task_id] = model.lora_layer.v_lora_B[task_id-1]
            
            for epoch in range(args.epochs): 
                train_stats = train_one_epoch(model=model, criterion=criterion, data_loader=data_loader[task_id]['train'], optimizer=optimizer, 
                                            device=device, epoch=epoch, task_dataset_map=task_dataset_map, max_norm=args.clip_grad, set_training_mode=True, task_id=task_id, 
                                            class_mask=class_mask, args=args, training_type='hide')
        
                if lr_scheduler:
                    lr_scheduler.step(epoch)
            _compute_mean(model=model, data_loader=data_loader_per_cls, device=device, task_id=task_id,
                        class_mask=class_mask[task_id], args=args)


            if task_id > 0:
                pre_ca_test_stats = evaluate_till_now(model=model, data_loader=data_loader,
                                                    device=device, task_id=task_id, class_mask=class_mask,
                                                    acc_matrix=acc_matrix, args=args, 
                                                    evaluate_type='hide', original_model=vanilla_model,
                                                    target_dataset_map=target_dataset_map, 
                                                    target_task_map=target_task_map, task_dataset_map=task_dataset_map)
                train_task_adaptive_prediction(model, args, device, class_mask, task_id)

        test_stats = evaluate_till_now(model=model, data_loader=data_loader,
                                       device=device, task_id=task_id, class_mask=class_mask,
                                       acc_matrix=acc_matrix, args=args, 
                                       evaluate_type='hide', original_model=vanilla_model,
                                       target_dataset_map=target_dataset_map, 
                                       target_task_map=target_task_map, task_dataset_map=task_dataset_map)
        
        if not args.eval and args.output_dir and utils.is_main_process():
            Path(os.path.join(args.output_dir, 'checkpoint')).mkdir(parents=True, exist_ok=True)

            checkpoint_path = os.path.join(args.output_dir, 'checkpoint/task{}_checkpoint.pth'.format(task_id + 1))
            state_dict = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'args': args,
            }
            if args.sched is not None and args.sched != 'constant':
                state_dict['lr_scheduler'] = lr_scheduler.state_dict()

            utils.save_on_master(state_dict, checkpoint_path)


def train_and_evaluate_mixture_model(model, model_without_ddp, vanilla_model, shared_model, criterion, data_loader, data_loader_per_cls, optimizer, lr_scheduler, device, class_mask, target_dataset_map, target_task_map, task_dataset_map, args):
    acc_matrix = np.zeros((args.num_tasks, args.num_tasks))
    pre_ca_acc_matrix = np.zeros((args.num_tasks, args.num_tasks))
    global cls_mean
    global cls_cov
    cls_mean = dict()
    cls_cov = dict()

    for task_id in range(args.num_tasks):
        if task_id > 0 and args.reinit_optimizer:
            base_params = [p for name, p in model_without_ddp.named_parameters() if 'lora' in name and p.requires_grad == True]
            base_fc_params = [p for name, p in model_without_ddp.named_parameters() if 'lora' not in name and p.requires_grad == True]
            base_params = {'params': base_params, 'lr': args.lr, 'weight_decay': args.weight_decay}
            base_fc_params = {'params': base_fc_params, 'lr': args.lr, 'weight_decay': args.weight_decay}
            network_params = [base_params, base_fc_params]
            optimizer = create_optimizer(args, network_params)
            if args.sched != 'constant':
                lr_scheduler, _ = create_scheduler(args, optimizer)
            elif args.sched == 'constant':
                lr_scheduler = None
        
        original_checkpoint_path = os.path.join(args.vanilla_model_output_dir, 'checkpoint/task{}_checkpoint.pth'.format(task_id + 1))
        if os.path.exists(original_checkpoint_path):
            print('Loading checkpoint from:', original_checkpoint_path)
            original_checkpoint = torch.load(original_checkpoint_path, map_location=device)
            vanilla_model.load_state_dict(original_checkpoint['model'], strict=True)
        else:
            print('No checkpoint found at:', original_checkpoint_path)
            return
        
        shared_checkpoint_path = os.path.join(args.shared_model_output_dir, 'checkpoint/task{}_checkpoint.pth'.format(args.num_tasks))
        if os.path.exists(shared_checkpoint_path):
            print('Loading checkpoint from:', shared_checkpoint_path)
            shared_checkpoint = torch.load(shared_checkpoint_path, map_location=device)
            ckpt = shared_checkpoint['model']
            head = [k for k in ckpt.keys() if 'head' in k]
            for k in head:
                del ckpt[k]
            shared_model.load_state_dict(ckpt, strict=False)
        else:
            print('No checkpoint found at:', shared_checkpoint_path)
            return
        
        if args.eval:
            mixture_checkpoint_path = os.path.join(args.output_dir, 'checkpoint/task{}_checkpoint.pth'.format(task_id + 1))
            if os.path.exists(mixture_checkpoint_path):
                print('Loading checkpoint from:', mixture_checkpoint_path)
                mixture_checkpoint = torch.load(mixture_checkpoint_path, map_location=device)
                model_without_ddp.load_state_dict(mixture_checkpoint['model'], strict=True)
        else:
            if task_id >= 0:
                # shared_model update attention
                #shared_model.update_attention(task_id=task_dataset_map[task_id], device=device)

                # model copy shared_model's backbone
                #_copy_backbone_parameters(model.module, shared_model, args)
                _copy_lora_parameters(model.module, shared_model, task_id, task_dataset_map[task_id], args)
                #_copy_fc_parameters(model.module, shared_model, args)

                # shared_model recovery from original backbone
                #_copy_backbone_parameters(shared_model, vanilla_model, args)
                    
            
            for epoch in range(args.epochs): 
                train_stats = train_one_epoch(model=model, criterion=criterion, data_loader=data_loader[task_id]['train'], optimizer=optimizer, 
                                            device=device, epoch=epoch, task_dataset_map=task_dataset_map, max_norm=args.clip_grad, set_training_mode=True, task_id=task_id, 
                                            class_mask=class_mask, args=args, training_type='mixture')
        
                if lr_scheduler:
                    lr_scheduler.step(epoch)
            _compute_mean(model=model, data_loader=data_loader_per_cls, device=device, task_id=task_id,
                        class_mask=class_mask[task_id], args=args)


            if task_id > 0:
                # pre_ca_test_stats = evaluate_till_now(model=shared_model, data_loader=data_loader,
                #                                       device=device, task_id=task_id, class_mask=class_mask,
                #                                       acc_matrix=acc_matrix, args=args, 
                #                                       evaluate_type='mixture', original_model=vanilla_model,
                #                                       target_dataset_map=target_dataset_map, 
                #                                       target_task_map=target_task_map, task_dataset_map=task_dataset_map, mixture_model=model)
                train_task_adaptive_prediction(model, args, device, class_mask, task_id)

        
            
        test_stats = evaluate_till_now(model=shared_model, data_loader=data_loader,
                                       device=device, task_id=task_id, class_mask=class_mask,
                                       acc_matrix=acc_matrix, args=args, 
                                       evaluate_type='mixture', original_model=vanilla_model,
                                       target_dataset_map=target_dataset_map, 
                                       target_task_map=target_task_map, task_dataset_map=task_dataset_map, mixture_model=model)
        
        # recovery model
        #_copy_backbone_parameters(model, vanilla_model, args)
        
        if not args.eval and args.output_dir and utils.is_main_process():
            Path(os.path.join(args.output_dir, 'checkpoint')).mkdir(parents=True, exist_ok=True)

            checkpoint_path = os.path.join(args.output_dir, 'checkpoint/task{}_checkpoint.pth'.format(task_id + 1))
            state_dict = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'args': args,
            }
            if args.sched is not None and args.sched != 'constant':
                state_dict['lr_scheduler'] = lr_scheduler.state_dict()

            utils.save_on_master(state_dict, checkpoint_path)

@torch.no_grad()
def _copy_backbone_parameters(model1, model2, args):
    params2 = {name: param for name, param in model2.named_parameters()}
    if args.freeze:
        # freeze args.freeze[blocks, patch_embed, cls_token] parameters
        for n, p in model1.named_parameters():
            if n.startswith(tuple(args.freeze)):
                p.copy_(params2[n])

@torch.no_grad()
def _copy_fc_parameters(model1, model2, args):
    params2 = {name: param for name, param in model2.named_parameters()}
    for n, p in model1.named_parameters():
        if 'head' in n:
            p.copy_(params2[n])

@torch.no_grad()
def _copy_lora_parameters(model1, model2, task_id, dataset_id, args):
    if task_id == 0:
        model1.lora_layer.k_lora_A[task_id] = model2.lora_layer.k_lora_A[dataset_id]
        model1.lora_layer.k_lora_B[task_id] = model2.lora_layer.k_lora_B[dataset_id]
        model1.lora_layer.v_lora_A[task_id] = model2.lora_layer.v_lora_A[dataset_id]
        model1.lora_layer.v_lora_B[task_id] = model2.lora_layer.v_lora_B[dataset_id]
    else:
        model1.lora_layer.k_lora_A[task_id] = 0.1 * model2.lora_layer.k_lora_A[dataset_id] + 0.9*model1.lora_layer.k_lora_A[task_id-1]
        model1.lora_layer.k_lora_B[task_id] = 0.1 * model2.lora_layer.k_lora_B[dataset_id] + 0.9*model1.lora_layer.k_lora_B[task_id-1]
        model1.lora_layer.v_lora_A[task_id] = 0.1 * model2.lora_layer.v_lora_A[dataset_id] + 0.9*model1.lora_layer.v_lora_A[task_id-1]
        model1.lora_layer.v_lora_B[task_id] = 0.1 * model2.lora_layer.v_lora_B[dataset_id] + 0.9*model1.lora_layer.v_lora_B[task_id-1]

