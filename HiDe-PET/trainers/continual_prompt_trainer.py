import torch
from timm.models import create_model
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
import time, datetime, os, sys, random, numpy as np
from datasets import build_continual_dataloader
import vits.hide_prompt_vision_transformer as hide_prompt_vision_transformer

def train(args):
    device = torch.device(args.device)
    data_loader, data_loader_per_cls, class_mask, target_task_map = build_continual_dataloader(args)
    
    print(f'Creating model: {args.model}')
    model = create_model(args.model,
                        pretrained=args.pretrained,
                        num_classes=args.nb_classes,
                        drop_rate=args.drop,
                        drop_path_rate=args.drop_path,
                        drop_block_rate=None,
                        prompt_length=args.length,
                        top_k=args.top_k,
                        head_type=args.head_type,
                        use_e_prompt=args.use_e_prompt,
                        e_prompt_layer_idx=args.e_prompt_layer_idx,
                        use_prefix_tune_for_e_prompt=args.use_prefix_tune_for_e_prompt,
                        prompt_type=args.prompt_type,
                        )
    model.to(device)
    if args.prompt_type == 'continual':
        from engines.continual_pet_engine import train_and_evaluate, evaluate_till_now
    if args.prompt_type == 'momentum':
        from engines.momentum_pet_engine import train_and_evaluate, evaluate_till_now
    
    if args.freeze:
        # freeze args.freeze[blocks, patch_embed, cls_token] parameters
        for n, p in model.named_parameters():
            if n.startswith(tuple(args.freeze)):
                p.requires_grad = False

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    if args.eval:
        acc_matrix = np.zeros((args.num_tasks, args.num_tasks))

        for task_id in range(args.num_tasks):
            checkpoint_path = os.path.join(args.output_dir, 'checkpoint/task{}_checkpoint.pth'.format(task_id + 1))
            if os.path.exists(checkpoint_path):
                print('Loading checkpoint from:', checkpoint_path)
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint['model'])
            else:
                print('No checkpoint found at:', checkpoint_path)
                return
            _ = evaluate_till_now(model, data_loader, device,
                                  task_id, class_mask, acc_matrix, args, )

        return

    if args.unscale_lr:
        global_batch_size = args.batch_size
    else:
        global_batch_size = args.batch_size * args.world_size
    args.lr = args.lr * global_batch_size / 256.0

    if args.prompt_type == 'continual' and args.continual_type == 'slow_learner':
        base_params = [p for name, p in model_without_ddp.named_parameters() if 'prompt' in name and p.requires_grad == True]
        base_fc_params = [p for name, p in model_without_ddp.named_parameters() if 'prompt' not in name and p.requires_grad == True]
        base_params = {'params': base_params, 'lr': args.lr * args.slow_lr, 'weight_decay': args.weight_decay}
        base_fc_params = {'params': base_fc_params, 'lr': args.lr, 'weight_decay': args.weight_decay}
        network_params = [base_params, base_fc_params]
        optimizer = create_optimizer(args, network_params)
    else:
        optimizer = create_optimizer(args, model_without_ddp)

    
    if args.sched != 'constant':
        lr_scheduler, _ = create_scheduler(args, optimizer)
    elif args.sched == 'constant':
        lr_scheduler = None

    criterion = torch.nn.CrossEntropyLoss().to(device)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    train_and_evaluate(model, model_without_ddp, criterion, data_loader, data_loader_per_cls, optimizer, lr_scheduler, device, class_mask, args, target_task_map)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Total training time: {total_time_str}")

