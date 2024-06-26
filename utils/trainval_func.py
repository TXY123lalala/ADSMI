import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3'
from network.get_network import GetNetwork
import torch
import torch.nn as nn
from configs.default import *
import torch.nn.functional as F
from tqdm import tqdm
import random
from torch.optim.lr_scheduler import _LRScheduler
from utils.utils import *

import sys
sys.path.append("/data3/xytan/code/") 
#import DropPos.util.misc as misc
from DropPos.util import misc as misc
#import util.misc as misc
from DropPos.util.misc import NativeScalerWithGradNormCount as NativeScaler
from DropPos.util.datasets import ImageListFolder
from DropPos.util.pos_embed import interpolate_pos_embed
#from DropPos.util import lr_sched as lr_sched
from DropPos.util import utils as utils
import math
from Fedavg_DropPos_ADA_类似MAE.utils import lr_sched as lr_sched


from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from skimage.transform import resize
from collections import defaultdict

import timm.optim.optim_factory as optim_factory
from typing import Iterable
import builtins

class PolyLR(_LRScheduler):
    def __init__(self, optimizer, max_iters, power=0.9, last_epoch=-1, min_lr=1e-6):
        self.power = power
        self.max_iters = max_iters  # avoid zero lr
        self.min_lr = min_lr
        super(PolyLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        return [ max( base_lr * ( 1 - self.last_epoch/self.max_iters )**self.power, self.min_lr)
                for base_lr in self.base_lrs]

def Shuffle_Batch_Data(data_in):
    len_total = len(data_in)
    idx_list = list(range(len_total))
    random.shuffle(idx_list)
    return data_in[idx_list]


def loglikeli(mu, logvar, y_samples):
    return (-(mu - y_samples)**2 /logvar.exp()-logvar).mean()#.sum(dim=1).mean(dim=0)

def epoch_site_train_aug(model: torch.nn.Module,
                    data_loader: Iterable,
                    optimizer: torch.optim.Optimizer,
                    epoch: int,
                    loss_scaler,
                    site_name,
                    log_ten,
                    args=None):
    if not misc.is_main_process():
        def print_pass(*args):
            pass
        builtins.print = print_pass

    model.train(True)
    metrics = defaultdict(list)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    #header = 'Epoch: [{}/{}]'.format(epoch, args.local_epochs)
    header = 'Epoch: [{}]'.format(epoch, args.local_epochs)
    print_freq = 2

    accum_iter = 1
    optimizer.zero_grad()

    if args.drop_pos_type in ['mae_pos_target', 'multi_task']:
        sigma = (1 - epoch / float(args.epochs)) * args.label_smoothing_sigma if args.sigma_decay else args.label_smoothing_sigma
        num_patches = (args.input_size // args.token_size) ** 2
        smooth = _get_label_smoothing_map(int(num_patches), sigma)

    #for data_iter_step, (batch, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
    temp_dataloader = enumerate(data_loader)
    semantic_distance_criterion = nn.MSELoss()
    print("start aug")
    for data_iter_step, (batch, dataset_idx, sample_idx) in temp_dataloader:
        if dataset_idx[0]!=0 and data_iter_step==len(data_loader)-1:
            continue
        torch.cuda.empty_cache()
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        batch, _ = batch
        images, bool_masked_pos = batch
        samples = images.cuda(non_blocking=True)
        bool_masked_pos = bool_masked_pos.cuda(non_blocking=True).flatten(1).to(torch.bool)   # (N, L)

        batch_size=16
        if dataset_idx[0]%2==0:
            _, batch_aug= next(iter(temp_dataloader))
            batch_aug, _ , _= batch_aug
            batch_aug, _ = batch_aug
            images_aug, bool_masked_pos_aug = batch_aug
            samples_aug = images_aug.cuda(non_blocking=True)
            bool_masked_pos_aug = bool_masked_pos_aug.cuda(non_blocking=True).flatten(1).to(torch.bool)   # (N, L)

            last_features, temp, mae_loss = model(samples_aug, img_mae=samples, mask_ratio=args.mask_ratio,
                                   pos_mask_ratio=args.pos_mask_ratio, smooth=smooth, aug=True)
            acc1, model_loss = temp
            optimizer.zero_grad()
            
            # Total loss & backward
            loss = model_loss + args.alpha1 * mae_loss

            loss_value = loss.item()
            loss /= accum_iter
            if loss_scaler is None:
                loss.backward()
                if args.clip_grad:
                    grad_norm = utils.clip_gradients(model, args.clip_grad)
                utils.cancel_gradients_last_layer(epoch, model, freeze_last_layer=0)
                torch.cuda.empty_cache()
                optimizer.step()
            else:
                loss_scaler.scale(loss).backward()
                if args.clip_grad:
                    loss_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                    grad_norm = utils.clip_gradients(model, args.clip_grad)
                utils.cancel_gradients_last_layer(epoch, model, freeze_last_layer=0)
                loss_scaler.step(optimizer)
                loss_scaler.update()

            if (data_iter_step + 1) % accum_iter == 0:
                optimizer.zero_grad()

            torch.cuda.synchronize()

            loss = loss.detach().cpu().numpy()
            log_ten.add_scalar(f'{site_name}_train_step_loss', loss, epoch*len(data_loader)+data_iter_step)
            metrics["Loss/train"].append(loss.item())

            lr = optimizer.param_groups[0]["lr"]
            metric_logger.update(loss=loss_value)

            if args.drop_pos_type != 'vanilla_mae':
                metric_logger.update(acc1=acc1)
            metric_logger.update(lr=lr)

            loss_value_reduce = misc.all_reduce_mean(loss_value)

            if (data_iter_step + 1) >= len(data_loader):
                break


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def epoch_site_train(model: torch.nn.Module,
                    data_loader: Iterable,
                    optimizer: torch.optim.Optimizer,
                    epoch: int,
                    loss_scaler,
                    site_name,
                    log_ten,
                    args=None):
    if not misc.is_main_process():
        def print_pass(*args):
            pass
        builtins.print = print_pass

    model.train(True)
    metrics = defaultdict(list)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    #header = 'Epoch: [{}/{}]'.format(epoch, args.local_epochs)
    header = 'Epoch: [{}]'.format(epoch, args.local_epochs)
    print_freq = 2

    accum_iter = 1
    optimizer.zero_grad()

    if args.drop_pos_type in ['mae_pos_target', 'multi_task']:
        sigma = (1 - epoch / float(args.epochs)) * args.label_smoothing_sigma if args.sigma_decay else args.label_smoothing_sigma
        num_patches = (args.input_size // args.token_size) ** 2
        smooth = _get_label_smoothing_map(int(num_patches), sigma)

    

    for data_iter_step, (batch, _, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        torch.cuda.empty_cache()
        it = len(data_loader) * epoch + data_iter_step
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        batch, _ = batch
        images, bool_masked_pos = batch

        samples = images.cuda(non_blocking=True)
        #device = torch.device('cuda')
        #samples = images.to(device)
        bool_masked_pos = bool_masked_pos.cuda(non_blocking=True).flatten(1).to(torch.bool)   # (N, L)

        with torch.cuda.amp.autocast(loss_scaler is not None):
            if args.drop_pos_type == 'vanilla_drop_pos':
                acc1, loss = model(samples, mask_ratio=args.mask_ratio)
            elif args.drop_pos_type == 'mae_pos_target':
                feature, temp = model(samples, mask_ratio=args.mask_ratio,
                                   pos_mask_ratio=args.pos_mask_ratio, smooth=smooth)
                acc1, loss = temp
            elif args.drop_pos_type == 'vanilla_mae':
                loss = model(samples, mask_ratio=args.mask_ratio)
            elif args.drop_pos_type == 'multi_task':
                acc1, loss_drop_pos, loss_mae = model(samples, mask_ratio=args.mask_ratio,
                                                      pos_mask_ratio=args.pos_mask_ratio,
                                                      smooth=smooth)
                loss = args.pos_weight * loss_drop_pos + loss_mae

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, skip".format(loss_value))
            sys.exit(1)

        loss /= accum_iter

        if loss_scaler is None:
            loss.backward()
            if args.clip_grad:
                grad_norm = utils.clip_gradients(model, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, model, freeze_last_layer=0)
            torch.cuda.empty_cache()
            optimizer.step()
        else:
            loss_scaler.scale(loss).backward()
            if args.clip_grad:
                loss_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                grad_norm = utils.clip_gradients(model, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, model, freeze_last_layer=0)
            loss_scaler.step(optimizer)
            loss_scaler.update()

        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        loss = loss.detach().cpu().numpy()
        log_ten.add_scalar(f'{site_name}_train_step_loss', loss, epoch*len(data_loader)+data_iter_step)
        metrics["Loss/train"].append(loss.item())

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss_value)
        if args.drop_pos_type == 'multi_task':
            metric_logger.update(mae=loss_mae.item())
            metric_logger.update(pos=loss_drop_pos.item())

        if args.drop_pos_type != 'vanilla_mae':
            metric_logger.update(acc1=acc1)
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)

        if (data_iter_step + 1) >= len(data_loader):
            break

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def _get_label_smoothing_map(num_patches, sigma):
    if sigma == 0.:
        # without label smoothing
        return torch.eye(num_patches)

    weight = torch.zeros([num_patches, num_patches])
    w = int(math.sqrt(num_patches))

    # for each patch i (0 to num_patches-1), its coordinate is (i // w, i % w)
    for i in range(num_patches):
        x_i, y_i = i // w, i % w
        for j in range(num_patches):
            x_j, y_j = j // w, j % w
            dist = (x_i - x_j) ** 2 + (y_i - y_j) ** 2
            weight[i, j] = math.exp(-dist / sigma ** 2)

    # normalize
    return weight / weight.sum(-1)
    
def site_train(comm_rounds, site_name, args, model, optimizer, scheduler, dataloader, dataloader_after, log_ten):
    tbar = tqdm(range(args.local_epochs))
    for local_epoch in tbar:
        tbar.set_description(f'{site_name}_train')
        #dataloader.sampler.set_epoch(local_epoch)
        epoch_site_train(model=model, data_loader=dataloader, optimizer=optimizer, epoch=comm_rounds*args.local_epochs + local_epoch, loss_scaler=None, site_name=site_name, log_ten=log_ten, args=args,)
        epoch_site_train_aug(model=model, data_loader=dataloader_after, optimizer=optimizer, epoch=comm_rounds*args.local_epochs + local_epoch, loss_scaler=None, site_name=site_name, log_ten=log_ten, args=args,)

def site_evaluation(epochs, site_name, args, model, dataloader, metric, judge, ret_samples_ids=None):
    
    model.eval()
    metric.reset()
    ret_samples = []
    print("evaluate")
    if judge=='train_single':
        directoryName = 'results_%s_%s'%(args.dataType, args.model)
        directoryName = os.path.join(directoryName, 'train_single', site_name)
    elif judge=='train_global':
        directoryName = 'results_%s_%s'%(args.dataType, args.model)
        directoryName = os.path.join(directoryName, 'train_global', site_name)
    else:
        directoryName = 'results_%s_%s'%(args.dataType, args.model)
        directoryName = os.path.join(directoryName, judge)

    #directoryName = os.path.join(directoryName, site_name)
    if args.save_val_results:
        if not os.path.exists(directoryName):
            os.mkdir(directoryName)
        denorm = Denormalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
        img_id = 0

    with torch.no_grad():

        for i, data_list in enumerate(dataloader):
            
            images, labels, domain_labels = data_list
            
            images = images.cuda()
            labels = labels.cuda()
            domain_labels = domain_labels.cuda()

            outputs = model(images)
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()
            
            metric.update(targets, preds)
            if ret_samples_ids is not None and i in ret_samples_ids:  # get vis samples
                ret_samples.append(
                    (images[0].detach().cpu().numpy(), targets[0], preds[0]))

            if args.save_val_results:
                for i in range(len(images)):
                    image = images[i].detach().cpu().numpy()
                    target = targets[i]
                    pred = preds[i]

                    image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
                    #target =(target).astype(np.uint8)
                    #pred = (pred).astype(np.uint8)
                    target =(target*255).astype(np.uint8)
                    pred = (pred*255).astype(np.uint8)

                    Image.fromarray(image).save(directoryName + '/%d_image.png' % img_id)
                    Image.fromarray(target).save(directoryName +'/%d_target.png' % img_id)
                    Image.fromarray(pred).save(directoryName+'/%d_pred.png' % img_id)

                    fig = plt.figure()
                    plt.imshow(image)
                    plt.axis('off')
                    plt.imshow(pred, alpha=0.7)
                    ax = plt.gca()
                    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    plt.savefig(directoryName+'/%d_overlay.png' % img_id, bbox_inches='tight', pad_inches=0)
                    plt.close()
                    img_id += 1

        score = metric.get_results()
        print("----------------")
        print(score['Mean IoU'])
        print("----------------")
    return score, ret_samples 


def site_evaluation_class_level(epochs, site_name, args, model, dataloader, log_file, log_ten, metric, note='after_fed'):
    model.eval()
    with torch.no_grad():
        for imgs, labels, domain_labels, in dataloader:
            imgs = imgs.cuda()
            output = model(imgs)
            metric.update(output, labels)
    results_dict = metric.results()
    log_ten.add_scalar(f'{note}_{site_name}_loss', results_dict['loss'], epochs)
    log_ten.add_scalar(f'{note}_{site_name}_acc', results_dict['acc'], epochs)
    log_ten.add_scalar(f'{note}_{site_name}_class_acc', results_dict['class_level_acc'], epochs)
    log_file.info(f'{note} Round: {epochs:3d} | Epochs: {args.local_epochs*epochs:3d} | Domain: {site_name} | loss: {results_dict["loss"]:.4f} | Acc: {results_dict["acc"]*100:.2f}% | C Acc: {results_dict["class_level_acc"]*100:.2f}%')

    return results_dict

def site_only_evaluation(model, dataloader, metric):
    model.eval()
    with torch.no_grad():
        for imgs, labels, domain_labels, in dataloader:
            imgs = imgs.cuda()
            output = model(imgs)
            metric.update(output, labels)
    results_dict = metric.results()
    return results_dict

def GetFedModel(args, num_classes, is_train=True):
    global_model = GetNetwork()
    #device = torch.device('cuda')
    #global_model = torch.nn.DataParallel(global_model, device_ids=device_ids).cuda(device_ids[0])
    #global_model = torch.nn.DataParallel(global_model)
    #global_model.to(device)
    global_model = global_model.cuda()
    #global_model = global_model.cpu()
    model_dict = {}
    optimizer_dict = {}
    scheduler_dict = {}
    
    if args.dataset == 'pacs':
        domain_list = pacs_domain_list
    elif args.dataset == 'officehome':
        domain_list = officehome_domain_list
    elif args.dataset == 'domainNet':
        domain_list = domainNet_domain_list
    elif args.dataset == 'terrainc':
        domain_list = terra_incognita_list
        
    for domain_name in domain_list:
        #model_dict[domain_name], _ = GetNetwork(args, num_classes, is_train)
        model_dict[domain_name] = GetNetwork()
        #model_dict[domain_name] = torch.nn.DataParallel(model_dict[domain_name], device_ids=device_ids).cuda(device_ids[0])
        #model_dict[domain_name] = torch.nn.DataParallel(model_dict[domain_name])
        #model_dict[domain_name].to(device)
        model_dict[domain_name] = model_dict[domain_name].cuda()

        eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

        if args.lr is None:  # only base_lr is specified
            args.lr = args.blr * eff_batch_size / 256

        print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
        print("actual lr: %.2e" % args.lr)

        print("accumulate grad iterations: %d" % args.accum_iter)
        print("effective batch size: %d" % eff_batch_size)
        
        """
        optimizer_dict[domain_name] = torch.optim.SGD(params=[
            {'params': model_dict[domain_name].backbone.parameters(), 'lr': 0.1*args.lr},
            {'params': model_dict[domain_name].classifier.parameters(), 'lr': args.lr},
        ], lr=args.lr, momentum=0.9, weight_decay=args.weight_decay) 
        """
        #optimizer_dict[domain_name] = torch.optim.Adam(model_dict[domain_name].parameters(), lr=args.learning_rate)
        param_groups = optim_factory.param_groups_weight_decay(model_dict[domain_name], args.weight_decay)
        optimizer_dict[domain_name] = torch.optim.AdamW(param_groups, lr=args.learning_rate, betas=(0.9, 0.95))
         
        total_epochs = args.local_epochs * args.comm
        """
        if args.lr_policy == 'step':
            scheduler_dict[domain_name] = torch.optim.lr_scheduler.StepLR(optimizer_dict[domain_name], step_size=int(total_epochs *0.8), gamma=0.1)
        elif args.lr_policy == 'mul_step':
            scheduler_dict[domain_name] = torch.optim.lr_scheduler.MultiStepLR(optimizer_dict[domain_name], milestones=[int(total_epochs*0.3), int(total_epochs*0.8)], gamma=0.1)
        elif args.lr_policy == 'exp95':
            scheduler_dict[domain_name] = torch.optim.lr_scheduler.ExponentialLR(optimizer_dict[domain_name], gamma=0.95)
        elif args.lr_policy == 'exp98':
            scheduler_dict[domain_name] = torch.optim.lr_scheduler.ExponentialLR(optimizer_dict[domain_name], gamma=0.98)
        elif args.lr_policy == 'exp99':
            scheduler_dict[domain_name] = torch.optim.lr_scheduler.ExponentialLR(optimizer_dict[domain_name], gamma=0.99)   
        elif args.lr_policy == 'cos':
            scheduler_dict[domain_name] = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_dict[domain_name], T_max=total_epochs)
        """
        scheduler_dict[domain_name] = PolyLR(optimizer_dict[domain_name], 30e3, power=0.9)     #???

        
            
    return global_model, model_dict, optimizer_dict, scheduler_dict

def SaveCheckPoint(args, model, epochs, path, optimizer=None, schedule=None, note='best_val'):
    check_dict = {'args':args, 'epochs':epochs, 'model':model.state_dict(), 'note': note}
    if optimizer is not None:
        check_dict['optimizer'] = optimizer.state_dict()
    if schedule is not None:
        check_dict['shceduler'] = schedule.state_dict()
    if not os.path.isdir(path):
        os.makedirs(path)
        
    torch.save(check_dict, os.path.join(path, note+'.pt'))
    

