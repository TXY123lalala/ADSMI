import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
import torch
#device = torch.device('cuda')
import sys
sys.path.append("/data/tyc/code/txy/Fedavg_DropPos_ADA_LTD")
import argparse
from network.get_network import GetNetwork
from torch.utils.tensorboard.writer import SummaryWriter
from data.pacs_dataset import PACS_FedDG
from utils.classification_metric import Classification 
from utils.log_utils import *
from utils.fed_merge import Cal_Weight_Dict, FedAvg, FedUpdate
from utils.trainval_func import site_evaluation, site_train, GetFedModel, SaveCheckPoint, _get_label_smoothing_map
import torch.nn.functional as F
from tqdm import tqdm
from metrics import StreamSegMetrics

from data.mask_transform import MaskTransform_ADA
from ADA_data import data_helper
from torch import nn
from torch import optim
import utils.lr_sched as lr_sched
from utils.utils import clip_gradients, cancel_gradients_last_layer
from utils.contrastive_loss import SupConLoss

def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='pacs', choices=['pacs'], help='Name of dataset')
    #parser.add_argument("--model", type=str, default='deeplabv3plus_resnet50',
    #                    choices=['deeplabv3plus_resnet50'], help='model name')
    parser.add_argument("--test_domain", type=str, default='C_1',
                        choices=['C_1', 'C_2', 'C_3', 'C_4', 'C_5', 'C_6', 'CVC-ClinicDB'], help='the domain name for testing')
    parser.add_argument('--num_classes', help='number of classes default 7', type=int, default=2)
    parser.add_argument('--batch_size', help='batch_size', type=int, default=16)
    parser.add_argument('--local_epochs', help='epochs number', type=int, default=1)
    parser.add_argument('--comm', help='epochs number', type=int, default=500)
    parser.add_argument('--learning_rate', help='learning rate', type=float, default=3e-4)
    parser.add_argument("--lr_policy", type=str, default='step', choices=['step'],
                        help="learning rate scheduler policy")
    parser.add_argument('--note', help='note of experimental settings', type=str, default='fedavg')
    parser.add_argument('--display', help='display in controller', action='store_true')
    parser.add_argument("--weight_decay", type=float, default=0.05,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])
    parser.add_argument("--dataType", type=str, default='polypGen',
                        help="path to Dataset")
    parser.add_argument("--save_val_results", action='store_true', default=True,
                        help="save segmentation results to \"./results_polypGen\"")
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    

    
    # DropPos parameters
    parser.add_argument('--drop_pos_type', type=str,
                        choices=['vanilla_mae',         # original MAE
                                 'mae_pos_target',      # DropPos with patch masking
                                 'multi_task'],         # DropPos with an auxiliary MAE loss
                        default='mae_pos_target')
    parser.add_argument('--mask_token_type', type=str,
                        choices=['param',       # learnable parameters
                                 'zeros',       # zeros
                                 'wrong_pos'],  # random wrong positions
                        default='param')
    parser.add_argument('--pos_mask_ratio', default=0.75, type=float,
                        help='Masking ratio of position embeddings.')
    parser.add_argument('--shuffle', action='store_true',
                        help='Shuffle before forward to encoder.')
    parser.set_defaults(shuffle=False)
    parser.add_argument('--pos_weight', type=float, default=1.,
                        help='Loss weight for position prediction when multi-task.')
    parser.add_argument('--label_smoothing_sigma', type=float, default=0.,
                        help='Label smoothing parameter for position prediction, 0 means no smoothing.')
    parser.add_argument('--sigma_decay', action='store_true',
                        help='Decay label smoothing sigma during training (linearly to 0).')
    parser.set_defaults(sigma_decay=False)
    parser.add_argument('--conf_ignore', action='store_true',
                        help='Ignore confident patches when computing objective.')
    parser.set_defaults(conf_ignore=False)
    parser.add_argument('--attn_guide', action='store_true',
                        help='Attention-guided loss weight.')
    parser.set_defaults(attn_guide=False)

    # Model parameters
    parser.add_argument('--model', default='DropPos_mae_vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int, help='images input size')
    parser.add_argument('--token_size', default=int(224 / 16), type=int,
                        help='number of patch (in one dimension), usually input_size//16')
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--norm_pix_loss', default=True,
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    
    # Mask generator (UM-MAE)
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--mask_block', action='store_true',
                        help='Block sampling for supporting pyramid-based vits')
    parser.set_defaults(mask_block=False)

    
    # Training/Optimization parameters
    
    parser.add_argument('--clip_grad', type=float, default=5., help="""Maximal parameter
            gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
            help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR, 40 for MAE and 10 for SimMIM')
    
    # augumentation
    parser.add_argument("--T_min", type=int, default=10, help="Number of iterations in Min-phase")
    parser.add_argument("--T_max", type=int, default=15, help="Number of iterations in Max-phase")
    #parser.add_argument("--current_iter", type=int, default=0, help="current iters")
    parser.add_argument("--gamma", type=float, default=1.0, help="Higher value leads to stricter distance constraint")
    parser.add_argument("--adv_learning_rate", type=float, default=1.0, help="Learning rate for adversarial training")
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')

    parser.add_argument("--alpha1", default=1, type=float)
    parser.add_argument("--alpha2", default=1, type=float)

    return parser.parse_args()

def loglikeli(mu, logvar, y_samples):
    return (-(mu - y_samples)**2 /logvar.exp()-logvar).mean()#.sum(dim=1).mean(dim=0)


def _do_max_phase(self, model, model_optimizer, T_max, X_n, smooth):
    # shape of X_n: B X H X W X C
    # from the Tensorflow implementation given by paper author, we can see N = BatchSize
    X_n = X_n.to(self.device)
    class_criterion = nn.CrossEntropyLoss()
    semantic_distance_criterion = nn.MSELoss()
    max_optimizer = optim.SGD([X_n.requires_grad_()], lr=self.adv_learning_rate)

    #model.eval()
    model.train()

    init_feature = None
    for i in range(T_max):
        #print("inner")
        #对增强数据优化
        max_optimizer.zero_grad()

        last_features, temp = model(X_n, mask_ratio=self.mask_ratio,
                                   pos_mask_ratio=self.pos_mask_ratio, smooth=smooth)
        acc1, loss = temp
        #loss_copy = loss
        #last_features, class_logit = self.model(X_n)
        if i == 0:
            init_feature = last_features.clone().detach()
        #_, cls_pred = class_logit.max(dim=1)
        #class_loss = class_criterion(class_logit, Y_n)
        data_model = torch.cat([init_feature, last_features])
        feature_loss = semantic_distance_criterion(data_model[self.batch_size:], data_model[:self.batch_size])
        
        #adv_loss = self.args.gamma * feature_loss - class_loss
        adv_loss = self.gamma * feature_loss - loss
        #print("feature_loss :%g , loss :%g" %(feature_loss,loss) )
        
        adv_loss.requires_grad_(True)
        adv_loss.backward(retain_graph=True)
        max_optimizer.step()

        #对分割模型优化
        last_features, temp = model(X_n, mask_ratio=self.mask_ratio,
                                   pos_mask_ratio=self.pos_mask_ratio, smooth=smooth)
        acc1, model_loss = temp
        model_optimizer.zero_grad()
        # Maximize MI between z and z_hat
        data_model = torch.cat([init_feature, last_features])
        #emb_src = F.normalize(data_model[:self.batch_size]).unsqueeze(1)
        #emb_aug = F.normalize(data_model[self.batch_size:]).unsqueeze(1)
        con_loss = semantic_distance_criterion(data_model[self.batch_size:], data_model[:self.batch_size])
        #con = SupConLoss()
        #con_loss = con(torch.cat([emb_src, emb_aug], dim=1))
        # Likelihood
        p_logvar = nn.ReLU()
        p_mu =  nn.LeakyReLU()
        mu = p_mu(data_model[:self.batch_size])
        logvar = p_logvar(data_model[:self.batch_size])
        #mu = tuple['mu'][class_l.size(0):]
        #logvar = tuple['logvar'][class_l.size(0):]
        y_samples = data_model[self.batch_size:]
        likeli = -loglikeli(mu, logvar, y_samples)
        # Total loss & backward
        model_loss = model_loss + self.alpha2*likeli + self.alpha1*con_loss
        model_loss.requires_grad_(True)
        model_loss.backward(retain_graph=True)
        model_optimizer.step()

        # 解除变量引用与实际值的指向关系
        del feature_loss, last_features
    #return X_n.to('cpu').detach(), cls_pred.to('cpu').detach().flatten().tolist(), adv_loss
    return X_n.to('cpu').detach(), adv_loss, model, model_optimizer


def main():
    '''log part'''
    file_name = 'fedavg_'+os.path.split(__file__)[1].replace('.py', '')
    args = get_argparse()
    log_dir, tensorboard_dir = Gen_Log_Dir(args, file_name=file_name)
    #print("---------log_dir----------")
    log_ten = SummaryWriter(log_dir=tensorboard_dir)
    log_file = Get_Logger(file_name=log_dir + 'train.log', display=args.display)
    Save_Hyperparameter(log_dir, args)
    
    '''dataset and dataloader'''
    dataobj = PACS_FedDG(args, test_domain=args.test_domain, batch_size=args.batch_size)
    dataloader_dict, dataset_dict = dataobj.GetData()
    
    '''model'''
    #metric = Classification()
    metric = StreamSegMetrics(args.num_classes)

    global_model, model_dict, optimizer_dict, scheduler_dict = GetFedModel(args, args.num_classes)

    for domain_name in dataobj.train_domain_list:
        K = int(2 * len(dataloader_dict[domain_name]['train']))
        #K=1
        for k in range(K):

            if args.drop_pos_type in ['mae_pos_target', 'multi_task']:
                sigma = (1 - k / float(args.K)) * args.label_smoothing_sigma if args.sigma_decay else args.label_smoothing_sigma
                num_patches = (args.input_size // args.token_size) ** 2
                smooth = _get_label_smoothing_map(int(num_patches), sigma)

            args.current_iter = 0
            class_loss, class_acc = 0.0, 0.0

            #for it, (batch, _) in enumerate(data_loader_train):
            
            for it, (batch, _) in enumerate(dataloader_dict[domain_name]['train']):
                batch, _ = batch
                data, bool_masked_pos = batch
                if domain_name == 'C_5' or domain_name == 'C_6':
                    if args.current_iter == 3:
                        print("Min-phase ended, class loss: %g, class acc: %g. %d-th Max-phase started!"
                            % (class_loss, class_acc, k))
                    
                        transform_ADA = MaskTransform_ADA(args)
                        data_adv, loss_adv, model_dict[domain_name], optimizer_dict[domain_name] = _do_max_phase(args, model_dict[domain_name],  optimizer_dict[domain_name], args.T_max, data, smooth)
                        #dataloader_dict[domain_name]['train'] = data_helper.append_adversarial_samples(
                        #args, dataloader_dict[domain_name]['train'], data, transform_ADA)
                        dataloader_dict[domain_name]['train'] = data_helper.append_adversarial_samples(
                            args, dataloader_dict[domain_name]['train'], data_adv, transform_ADA)
                        
                        print("%d-th Max-phase ended, Adv loss: %g" %(k, -1 * loss_adv))
                        break
                if args.current_iter == args.T_min:
                    print("Min-phase ended, class loss: %g, class acc: %g. %d-th Max-phase started!"
                            % (class_loss, class_acc, k))
                    
                    transform_ADA = MaskTransform_ADA(args)
                    data_adv, loss_adv, model_dict[domain_name], optimizer_dict[domain_name] = _do_max_phase(args, model_dict[domain_name], optimizer_dict[domain_name], args.T_max, data, smooth)
                    #dataloader_dict[domain_name]['train'] = data_helper.append_adversarial_samples(
                    #    args, dataloader_dict[domain_name]['train'], data, transform_ADA)
                    dataloader_dict[domain_name]['train'] = data_helper.append_adversarial_samples(
                        args, dataloader_dict[domain_name]['train'], data_adv, transform_ADA)
                    
                    print("%d-th Max-phase ended, Adv loss: %g" %(k, -1 * loss_adv))
                    break

                accum_iter = 1
                if it % accum_iter == 0:
                    lr_sched.adjust_learning_rate(optimizer_dict[domain_name], it / len(dataloader_dict[domain_name]['train']) + k, args)
                model_dict[domain_name].train()
                samples = data.cuda(non_blocking=True)
                bool_masked_pos = bool_masked_pos.cuda(non_blocking=True).flatten(1).to(torch.bool)   # (N, L)
                feature, temp = model_dict[domain_name](samples, mask_ratio=args.mask_ratio,
                                    pos_mask_ratio=args.pos_mask_ratio, smooth=smooth)
                acc1, loss = temp
                
                loss /= accum_iter

                loss.backward()
                if args.clip_grad:
                    grad_norm = clip_gradients(model_dict[domain_name], args.clip_grad)
                cancel_gradients_last_layer(k, model_dict[domain_name], freeze_last_layer=0)
                optimizer_dict[domain_name].step()
                optimizer_dict[domain_name].zero_grad()
                lr = optimizer_dict[domain_name].param_groups[0]["lr"]  

                args.current_iter = args.current_iter + 1

    weight_dict = Cal_Weight_Dict(dataset_dict, site_list=dataobj.train_domain_list)
    #FedUpdate(model_dict, global_model)
    best_val = 0.
    for i in range(args.comm+1):
        FedUpdate(model_dict, global_model)
        for domain_name in dataobj.train_domain_list:
            site_train(i, domain_name, args, model_dict[domain_name], optimizer_dict[domain_name], 
                       scheduler_dict[domain_name],dataloader_dict[domain_name]['train'], log_ten)
            
        FedAvg(model_dict, weight_dict, global_model)

        if i%50==0 or i==269:
            SaveCheckPoint(args, global_model, args.comm, os.path.join(log_dir, 'checkpoints'), note=f'{i}_global_model')
            for domain_name in dataobj.train_domain_list: 
                SaveCheckPoint(args, model_dict[domain_name], args.comm, os.path.join(log_dir, 'checkpoints'), note=f'{i}_{domain_name}_model')
        
        
    SaveCheckPoint(args, global_model, args.comm, os.path.join(log_dir, 'checkpoints'), note='global_model')
    for domain_name in dataobj.train_domain_list: 
        SaveCheckPoint(args, model_dict[domain_name], args.comm, os.path.join(log_dir, 'checkpoints'), note=f'last_{domain_name}_model')
    
if __name__ == '__main__':
    main()