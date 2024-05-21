# set up environment
import numpy as np
import random 
import datetime
import logging
import matplotlib.pyplot as plt
import os

from models.model_single import ModelEmb
from models.unet import UNet
from monai.losses import FocalLoss

from utils.Losses import FocalDiceLoss

join = os.path.join
from tqdm import tqdm
from torch.backends import cudnn
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.nn as nn
import torchio as tio
from torch.utils.data.distributed import DistributedSampler
from segment_anything.build_sam3D import sam_model_registry3D
import argparse
from torch.cuda import amp
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from monai.losses import DiceCELoss
from contextlib import nullcontext
from utils.click_method import get_next_click3D_torch_2
from utils.data_loader import Dataset_Union_ALL, Union_Dataloader
from utils.data_paths import img_datas
import torch.nn.functional as F
from sklearn.model_selection import ParameterGrid
from monai.networks.nets import SwinUNETR


# set random seed
def set_seed(seed_value):
    """Set seed for reproducibility."""
    random.seed(seed_value)  # Python random module
    np.random.seed(seed_value)  # NumPy
    torch.manual_seed(seed_value)  # PyTorch
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

seed_value = 42
set_seed(seed_value)

# %% set up parser
parser = argparse.ArgumentParser()
parser.add_argument('--task_name', type=str, default='picai_empty_included')
parser.add_argument('--click_type', type=str, default='random')
parser.add_argument('--multi_click', action='store_true', default=False)
parser.add_argument('--model_type', type=str, default='vit_b_ori')
parser.add_argument('--checkpoint', type=str, default='./ckpt/sam_med3d_turbo.pth')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--work_dir', type=str, default='work_dir')

# train
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--gpu_ids', type=int, nargs='+', default=[1,2,3])
parser.add_argument('--multi_gpu', action='store_true', default=True)
parser.add_argument('--resume', action='store_true', default=False)
parser.add_argument('--allow_partial_weight', action='store_true', default=False)
parser.add_argument('--hack_prompt', type=bool, default=True)
parser.add_argument('--loss_fn', type=str, default='focal_dice')
parser.add_argument('--MODEL_SAVE_PATH', type=str, default='./workdir')

# lr_scheduler and optimizer
parser.add_argument('--lr_scheduler', type=str, default='multisteplr')
parser.add_argument('--step_size', type=list, default=[1, 2])
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--num_epochs', type=int, default=200)
parser.add_argument('--img_size', type=int, default=128)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--accumulation_steps', type=int, default=20)
parser.add_argument('--lr', type=float, default=8e-4)
parser.add_argument('--weight_decay', type=float, default=0.01)
parser.add_argument('--beta', type=float, default=0.9)
parser.add_argument('--port', type=int, default=12361)

# prompt generator
parser.add_argument('-depth_wise', '--depth_wise', default=False, help='image size', required=False)
parser.add_argument('-order', '--order', default=85, help='image size', required=False)

args = parser.parse_args()
args.lr = 1e-3
task_name = 'picai_dice_ce'
args.num_epochs = 50
args.accumulation_steps = 1
args.hack_prompt = True
args.loss_fn = 'dice_ce'
args.gpu_ids = [2, 3]
args.batch_size = 1
args.multi_gpu = False

grid_search_params = {
    "weight_decay": [0.1, 0.01],
    "lr": [1e-3, 1e-4, 1e-5],
    "beta": [0.9, 0.8, 0.7]
}

param_grid = ParameterGrid(grid_search_params)


device = args.device
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in args.gpu_ids])
logger = logging.getLogger(__name__)
args.MODEL_SAVE_PATH = join(args.work_dir, args.task_name)
click_methods = {
    'random': get_next_click3D_torch_2,
}


def build_model(args):
    freeze = (False, True, False, False)
    sam_model = sam_model_registry3D[args.model_type](checkpoint=args.checkpoint, freeze=freeze[:3]).to(device)
    prompt_model = ModelEmb(args=args, freeze=freeze[3]).to(device)
    # prompt_model = UNet(spatial_dims=3,
    #         in_channels=1,
    #         out_channels=384,
    #         strides=[(2, 2, 2), (1, 2, 2), (1, 2, 2), (1, 2, 2), (2, 2, 2)],
    #         channels=[32, 64, 128, 256, 512, 1024]).to(device)
    if args.multi_gpu:
        if freeze[:3]==(True, True, True):
            # sam_model = DDP(sam_model, device_ids=[args.rank], output_device=args.rank, find_unused_parameters=True)
            prompt_model = DDP(prompt_model, device_ids=[args.rank], output_device=args.rank,
                               find_unused_parameters=True)
        elif freeze[3]==(True):
            sam_model = DDP(sam_model, device_ids=[args.rank], output_device=args.rank, find_unused_parameters=True)
        else:
            sam_model = DDP(sam_model, device_ids=[args.rank], output_device=args.rank, find_unused_parameters=True)
            prompt_model = DDP(prompt_model, device_ids=[args.rank], output_device=args.rank, find_unused_parameters=True)
    return sam_model, prompt_model


def get_dataloaders(args):
    train_dataset = Dataset_Union_ALL(paths=img_datas, transform=tio.Compose([
        tio.ToCanonical(),
        tio.CropOrPad(mask_name='label', target_shape=(args.img_size,args.img_size,args.img_size)), # crop only object region
        tio.RandomFlip(axes=(0, 1, 2)),
    ]),
    threshold=1000)

    if args.multi_gpu:
        train_sampler = DistributedSampler(train_dataset)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    train_dataloader = Union_Dataloader(
        dataset=train_dataset,
        sampler=train_sampler,
        batch_size=args.batch_size, 
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return train_dataloader

class BaseTrainer:
    def __init__(self, model, prompt_model, dataloaders, args):

        self.model = model
        self.prompt_model = prompt_model
        self.dataloaders = dataloaders
        self.args = args
        self.best_loss = np.inf
        self.best_dice = 0.0
        self.step_best_loss = np.inf
        self.step_best_dice = 0.0
        self.losses = []
        self.dices = []
        self.ious = []
        self.set_loss_fn()
        self.set_optimizer()
        self.set_lr_scheduler()
        if(args.resume):
            self.init_checkpoint(join(self.args.work_dir, self.args.task_name, 'sam_model_latest.pth'))
        else:
            self.init_checkpoint(self.args.checkpoint)

        self.norm_transform = tio.ZNormalization(masking_method=lambda x: x > 0)
        
    def set_loss_fn(self):
        if self.args.loss_fn == 'dice_ce':
            self.seg_loss = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
        elif self.args.loss_fn == 'focal':
            self.seg_loss = FocalLoss(alpha=0.95, reduction='mean')
        elif self.args.loss_fn == 'focal_dice':
            self.seg_loss = FocalDiceLoss(alpha=0.95)
    
    def set_optimizer(self):
        if hasattr(self.model, 'module'):
            sam_model = self.model.module
        else:
            sam_model = self.model
        if hasattr(self.prompt_model, 'module'):
            prompt_model = self.prompt_model.module
        else:
            prompt_model = self.prompt_model
        if self.args.hack_prompt:
            self.optimizer = torch.optim.AdamW([
                {'params': sam_model.image_encoder.parameters(), 'lr': self.args.lr * 0.1},
                {'params': sam_model.prompt_encoder.parameters() , 'lr': self.args.lr * 0.1},
                {'params': sam_model.mask_decoder.parameters(), 'lr': self.args.lr * 0.1},
                {'params': prompt_model.parameters(), 'lr': self.args.lr * 0.1}
            ], lr=self.args.lr, betas=(self.args.beta, 0.999), weight_decay=self.args.weight_decay)
        else:
            self.optimizer = torch.optim.AdamW([
                {'params': sam_model.image_encoder.parameters(),  'lr': self.args.lr * 0.1},
                {'params': sam_model.prompt_encoder.parameters(), 'lr': self.args.lr * 0.1},
                {'params': sam_model.mask_decoder.parameters(), 'lr': self.args.lr * 0.1},
            ], lr=self.args.lr, betas=(self.args.beta, 0.999), weight_decay=self.args.weight_decay)

    def set_lr_scheduler(self):
        if self.args.lr_scheduler == "multisteplr":
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                                self.args.step_size,
                                                                self.args.gamma)
        elif self.args.lr_scheduler == "steplr":
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                                self.args.step_size[0],
                                                                self.args.gamma)
        elif self.args.lr_scheduler == 'coswarm':
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer)
        else:
            self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, 0.1)

    def init_checkpoint(self, ckp_path):
        last_ckpt = None
        if os.path.exists(ckp_path):
            if self.args.multi_gpu:
                dist.barrier()
                last_ckpt = torch.load(ckp_path, map_location=self.args.device)
            else:
                last_ckpt = torch.load(ckp_path, map_location=self.args.device)
        
        if last_ckpt:
            if(self.args.allow_partial_weight):
                if hasattr(self.model, 'module'):
                    self.model.module.load_state_dict(last_ckpt['model_state_dict'], strict=False)
                else:
                    self.model.load_state_dict(last_ckpt['model_state_dict'], strict=False)
            else:
                if hasattr(self.model, 'module'):
                    self.model.module.load_state_dict(last_ckpt['model_state_dict'])
                else:
                    self.model.load_state_dict(last_ckpt['model_state_dict'])
            if not self.args.resume:
                self.start_epoch = 0 
            else:
                self.start_epoch = last_ckpt['epoch']
                self.optimizer.load_state_dict(last_ckpt['optimizer_state_dict'])
                self.lr_scheduler.load_state_dict(last_ckpt['lr_scheduler_state_dict'])
                self.losses = last_ckpt['losses']
                self.dices = last_ckpt['dices']
                self.best_loss = last_ckpt['best_loss']
                self.best_dice = last_ckpt['best_dice']
            print(f"Loaded checkpoint from {ckp_path} (epoch {self.start_epoch})")
        else:
            self.start_epoch = 0
            print(f"No checkpoint found at {ckp_path}, start training from scratch")

    def save_checkpoint(self, epoch, state_dict, model_name, describe="last"):
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": state_dict,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
            "losses": self.losses,
            "dices": self.dices,
            "best_loss": self.best_loss,
            "best_dice": self.best_dice,
            "args": self.args,
            "used_datas": img_datas,
        }, join(self.args.MODEL_SAVE_PATH, f"{model_name}_{describe}.pth"))

    def batch_auto_forward(self, sam_model, image_embedding, dense_embeddings, gt3D):

        sparse_embeddings_none, dense_embeddings_none = sam_model.prompt_encoder(
            points=None,
            boxes=None,
            masks=None,
        )
        low_res_masks, iou_predictions = sam_model.mask_decoder(
            image_embeddings=image_embedding.to(device),  # (B, 256, 64, 64)
            image_pe=sam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings_none,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )
        prev_masks = F.interpolate(low_res_masks, size=gt3D.shape[-3:], mode='trilinear', align_corners=False)
        return low_res_masks, prev_masks

    def batch_forward(self, sam_model, image_embedding, gt3D, low_res_masks, points=None):
        
        sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
            points=points,
            boxes=None,
            masks=low_res_masks,
        )
        low_res_masks, iou_predictions = sam_model.mask_decoder(
            image_embeddings=image_embedding.to(device), # (B, 256, 64, 64)
            image_pe=sam_model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            multimask_output=False,
        )
        prev_masks = F.interpolate(low_res_masks, size=gt3D.shape[-3:], mode='trilinear', align_corners=False)
        return low_res_masks, prev_masks

    def get_points(self, prev_masks, gt3D):
        batch_points, batch_labels = click_methods[self.args.click_type](prev_masks, gt3D)

        points_co = torch.cat(batch_points, dim=0).to(device)
        points_la = torch.cat(batch_labels, dim=0).to(device)

        self.click_points.append(points_co)
        self.click_labels.append(points_la)

        points_multi = torch.cat(self.click_points, dim=1).to(device)
        labels_multi = torch.cat(self.click_labels, dim=1).to(device)

        if self.args.multi_click:
            points_input = points_multi
            labels_input = labels_multi
        else:
            points_input = points_co
            labels_input = points_la
        return points_input, labels_input

    def auto_decode(self, sam_model, image_embedding, dense_embeddings, gt3D):
        return_loss = 0
        low_res_masks, prev_masks = self.batch_auto_forward(sam_model, image_embedding, dense_embeddings, gt3D)
        # # make masks one-hot encoded
        # gt3D = F.one_hot(gt3D.squeeze(1), num_classes=2)
        # gt3D = gt3D.permute(0, 4, 1, 2, 3).contiguous()
        # complementary_logits = -prev_masks
        # logits_tensor = torch.cat((complementary_logits, prev_masks), dim=1)
        # loss = self.seg_loss(logits_tensor, gt3D.to(torch.float32))
        loss = self.seg_loss(prev_masks, gt3D)
        return_loss += loss
        return prev_masks, return_loss

    def interaction(self, sam_model, image_embedding, gt3D, num_clicks):
        return_loss = 0
        prev_masks = torch.zeros_like(gt3D).to(gt3D.device)
        low_res_masks = F.interpolate(prev_masks.float(), size=(args.img_size//4,args.img_size//4,args.img_size//4))
        random_insert = np.random.randint(2, 9)
        for num_click in range(num_clicks):
            points_input, labels_input = self.get_points(prev_masks, gt3D)

            if num_click == random_insert or num_click == num_clicks - 1:
                low_res_masks, prev_masks = self.batch_forward(sam_model, image_embedding, gt3D, low_res_masks, points=None)
            else:
                low_res_masks, prev_masks = self.batch_forward(sam_model, image_embedding, gt3D, low_res_masks, points=[points_input, labels_input])
            loss = self.seg_loss(prev_masks, gt3D)
            return_loss += loss
        return prev_masks, return_loss
    
    def get_dice_score(self, prev_masks, gt3D):
        def compute_dice(mask_pred, mask_gt):
            mask_threshold = 0.5

            mask_pred = (mask_pred > mask_threshold)
            mask_gt = (mask_gt > 0)
            
            volume_sum = mask_gt.sum() + mask_pred.sum()

            volume_intersect = (mask_gt & mask_pred).sum()
            return (2*volume_intersect + 1e-5) / (volume_sum + 1e-5)
    
        pred_masks = (prev_masks > 0.5)
        true_masks = (gt3D > 0)
        dice_list = []
        for i in range(true_masks.shape[0]):
            dice_list.append(compute_dice(pred_masks[i], true_masks[i]))
        return (sum(dice_list)/len(dice_list)).item() 


    def train_epoch(self, epoch, num_clicks):
        epoch_loss = 0
        epoch_iou = 0
        epoch_dice = 0
        self.model.train()
        if hasattr(self.model, 'module'):
            sam_model = self.model.module
        else:
            sam_model = self.model
            # self.args.rank = -1

        if hasattr(self.prompt_model, 'module'):
            prompt_model = self.prompt_model.module
        else:
            prompt_model = self.prompt_model
            # self.args.rank = -1

        if not self.args.multi_gpu:
            self.args.rank = -1

        if not self.args.multi_gpu or (self.args.multi_gpu and self.args.rank == 0):
            tbar = tqdm(self.dataloaders)
        else:
            tbar = self.dataloaders

        self.optimizer.zero_grad()
        step_loss = 0
        volumes = []
        dices = []
        for step, (image3D, gt3D) in enumerate(tbar):

            my_context1 = self.model.no_sync if hasattr(self.model, 'module') and step % self.args.accumulation_steps != 0 else nullcontext
            my_context2 = self.prompt_model.no_sync if hasattr(self.prompt_model, 'module') and step % self.args.accumulation_steps != 0 else nullcontext

            with my_context1():
                with my_context2():

                    image3D = self.norm_transform(image3D.squeeze(dim=1)) # (N, C, W, H, D)
                    image3D = image3D.unsqueeze(dim=1)

                    image3D = image3D.to(device)
                    gt3D = gt3D.to(device).type(torch.long)
                    with amp.autocast():
                        if self.args.hack_prompt:
                            image_embedding = sam_model.image_encoder(image3D)
                            dense_embeddings = prompt_model(image3D)
                            prev_masks, loss = self.auto_decode(sam_model, image_embedding, dense_embeddings,  gt3D)
                            pred_list = []
                        else:
                            image_embedding = sam_model.image_encoder(image3D)

                            self.click_points = []
                            self.click_labels = []

                            pred_list = []

                            prev_masks, loss = self.interaction(sam_model, image_embedding, gt3D, num_clicks=11)
                    pred_mask = prev_masks.clone().detach()
                    epoch_dice += self.get_dice_score(torch.sigmoid(pred_mask), gt3D)

                    epoch_loss += loss.item()

                    cur_loss = loss.item()

                    loss /= self.args.accumulation_steps

                    self.scaler.scale(loss).backward()

            if step % self.args.accumulation_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                step_loss += cur_loss
                print_loss = step_loss / self.args.accumulation_steps
                step_loss = 0
                print_dice = self.get_dice_score(torch.sigmoid(pred_mask), gt3D)
            else:
                step_loss += cur_loss

            if not self.args.multi_gpu or (self.args.multi_gpu and self.args.rank == 0):
                if step % self.args.accumulation_steps == 0:
                    # target_volume = gt3D.sum().detach().cpu()
                    # volumes.append(target_volume)
                    # dices.append(print_dice.detach().cpu())
                    print(f'Epoch: {epoch}, Step: {step}, Loss: {print_loss}, Dice: {print_dice}')
                    if print_dice > self.step_best_dice:
                        self.step_best_dice = print_dice
                        if print_dice > 0.9:
                            self.save_checkpoint(
                                epoch,
                                sam_model.state_dict(),
                                model_name='sam_model',
                                describe=f'{epoch}_step_dice:{print_dice}_best'
                            )
                            self.save_checkpoint(
                                epoch,
                                prompt_model.state_dict(),
                                model_name='prompt_model',
                                describe=f'{epoch}_step_dice:{print_dice}_best'
                            )
                    if print_loss < self.step_best_loss:
                        self.step_best_loss = print_loss
            
        epoch_loss /= step
        epoch_dice /= step

        return epoch_loss, epoch_iou, epoch_dice, pred_list

    def eval_epoch(self, epoch, num_clicks):
        return 0
    
    def plot_result(self, plot_data, description, save_name):
        plt.plot(plot_data)
        plt.title(description)
        plt.xlabel('Epoch')
        plt.ylabel(f'{save_name}')
        plt.savefig(join(self.args.MODEL_SAVE_PATH, f'{save_name}.png'))
        plt.close()


    def train(self):
        self.scaler = amp.GradScaler()
        for epoch in range(self.start_epoch, self.args.num_epochs):
            print(f'Epoch: {epoch}/{self.args.num_epochs - 1}')

            if self.args.multi_gpu:
                dist.barrier()
                self.dataloaders.sampler.set_epoch(epoch)
            num_clicks = np.random.randint(1, 21)
            epoch_loss, epoch_iou, epoch_dice, pred_list = self.train_epoch(epoch, num_clicks)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            if self.args.multi_gpu:
                dist.barrier()
        
            if not self.args.multi_gpu or (self.args.multi_gpu and self.args.rank == 0):
                self.losses.append(epoch_loss)
                self.dices.append(epoch_dice)
                print(f'EPOCH: {epoch}, Loss: {epoch_loss}')
                print(f'EPOCH: {epoch}, Dice: {epoch_dice}')
                logger.info(f'Epoch\t {epoch}\t : loss: {epoch_loss}, dice: {epoch_dice}')

                if hasattr(self.model, 'module'):
                    sam_state_dict = self.model.module.state_dict()
                else:
                    sam_state_dict = self.model.state_dict()

                if hasattr(self.prompt_model, 'module'):
                    prompt_state_dict = self.prompt_model.module.state_dict()
                else:
                    prompt_state_dict = self.prompt_model.state_dict()

                
                # save latest checkpoint
                self.save_checkpoint(
                    epoch, 
                    sam_state_dict,
                    model_name='sam_model',
                    describe='latest'
                )

                self.save_checkpoint(
                    epoch,
                    prompt_state_dict,
                    model_name='prompt_model',
                    describe='latest'
                )

                # save train loss best checkpoint
                if epoch_loss < self.best_loss: 
                    self.best_loss = epoch_loss
                    self.save_checkpoint(
                        epoch,
                        sam_state_dict,
                        model_name='sam_model',
                        describe='loss_best'
                    )
                    self.save_checkpoint(
                        epoch,
                        prompt_state_dict,
                        model_name='prompt_model',
                        describe='loss_best'
                    )
                
                # save train dice best checkpoint
                if epoch_dice > self.best_dice: 
                    self.best_dice = epoch_dice
                    self.save_checkpoint(
                        epoch,
                        sam_state_dict,
                        model_name='sam_model',
                        describe='dice_best'
                    )
                    self.save_checkpoint(
                        epoch,
                        prompt_state_dict,
                        model_name='prompt_model',
                        describe='dice_best'
                    )

                self.plot_result(self.losses, 'Dice + Cross Entropy Loss', 'Loss')
                self.plot_result(self.dices, 'Dice', 'Dice')
        logger.info('=====================================================================')
        logger.info(f'Best loss: {self.best_loss}')
        logger.info(f'Best dice: {self.best_dice}')
        logger.info(f'Total loss: {self.losses}')
        logger.info(f'Total dice: {self.dices}')
        logger.info('=====================================================================')
        logger.info(f'args : {self.args}')
        logger.info(f'Used datasets : {img_datas}')
        logger.info('=====================================================================')

def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True
        
def device_config(args):
    try:
        if not args.multi_gpu:
            # Single GPU
            if args.device == 'mps':
                args.device = torch.device('mps')
            else:
                args.device = torch.device(f"cuda:{args.gpu_ids[0]}")
        else:
            args.nodes = 1
            args.ngpus_per_node = len(args.gpu_ids)
            args.world_size = args.nodes * args.ngpus_per_node

    except RuntimeError as e:
        print(e)

def get_dice_score(prev_masks, gt3D):
    def compute_dice(mask_pred, mask_gt):
        mask_threshold = 0.5
        mask_pred = (mask_pred > mask_threshold)
        mask_gt = (mask_gt > 0)
        volume_sum = mask_gt.sum() + mask_pred.sum()
        volume_intersect = (mask_gt & mask_pred).sum()
        return (2 * volume_intersect + 1e-5) / (volume_sum + 1e-5)
    pred_masks = (prev_masks > 0.5)
    true_masks = (gt3D > 0)
    dice_list = []
    for i in range(true_masks.shape[0]):
        dice_list.append(compute_dice(pred_masks[i], true_masks[i]))
    return (sum(dice_list) / len(dice_list)).item()
def train_one_epoch(epoch, model, dataloaders, optimizer, norm_transform, criterion, scaler):
    epoch_loss = 0
    epoch_iou = 0
    epoch_dice = 0
    model.train()

    tbar = dataloaders

    optimizer.zero_grad()
    step_loss = 0
    volumes = []
    dices = []
    step_best_dice = 0
    step_best_loss = np.inf
    for step, (image3D, gt3D) in enumerate(tbar):

        my_context1 = nullcontext
        my_context2 = nullcontext

        with my_context1():
            with my_context2():

                image3D = norm_transform(image3D.squeeze(dim=1))  # (N, C, W, H, D)
                image3D = image3D.unsqueeze(dim=1)

                image3D = image3D.to(device)
                gt3D = gt3D.to(device).type(torch.long)
                with amp.autocast():
                    pred = model(image3D)
                    loss = criterion(pred, gt3D)
                pred_mask = pred.clone().detach()
                epoch_dice += get_dice_score(torch.sigmoid(pred_mask), gt3D)

                epoch_loss += loss.item()

                cur_loss = loss.item()

                scaler.scale(loss).backward()

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        step_loss += cur_loss
        print_loss = step_loss
        step_loss = 0
        print_dice = get_dice_score(torch.sigmoid(pred_mask), gt3D)

        print(f'Epoch: {epoch}, Step: {step}, Loss: {print_loss}, Dice: {print_dice}')
        if print_dice > step_best_dice:
            step_best_dice = print_dice
        if print_loss < step_best_loss:
            step_best_loss = print_loss

    epoch_loss /= step
    epoch_dice /= step
    return epoch_loss, epoch_dice
def plot_result(self, plot_data, description, save_name):
    plt.plot(plot_data)
    plt.title(description)
    plt.xlabel('Epoch')
    plt.ylabel(f'{save_name}')
    plt.savefig(join('./', f'{save_name}.png'))
    plt.close()

def main():
    for params in param_grid:
        global task_name
        args.task_name = task_name
        for param_name, param in params.items():
            exec(f'args.{param_name} = {param}')
            args.task_name += f'_{param_name}_{param}'

        args.MODEL_SAVE_PATH = join(args.work_dir, args.task_name)
        os.makedirs(args.MODEL_SAVE_PATH, exist_ok=True)
        mp.set_sharing_strategy('file_system')
        device_config(args)
        if args.multi_gpu:
            mp.spawn(
                main_worker,
                nprocs=args.world_size,
                args=(args, )
            )
        else:
            random.seed(2023)
            np.random.seed(2023)
            torch.manual_seed(2023)
            # Load datasets
            dataloaders = get_dataloaders(args)
            # Build model
            # model = build_model(args)
            model = SwinUNETR(
            img_size=(128, 128, 128),
            in_channels=1,
            out_channels=1,
            feature_size=48,
            use_checkpoint=False,
        ).to(device)
            # # Create trainer
            # trainer = BaseTrainer(model, dataloaders, args)
            # # Train
            # trainer.train()
            norm_transform = tio.ZNormalization(masking_method=lambda x: x > 0)
            scaler = amp.GradScaler()
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                                [1, 2],
                                                                0.1)
            criterion = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
            losses = []
            dices = []
            best_dice = 0

            for epoch in range(args.num_epochs):
                print(f'Epoch: {epoch}/{args.num_epochs - 1}')

                epoch_loss, epoch_dice = train_one_epoch(epoch, model, dataloaders, optimizer, norm_transform, criterion, scaler)

                lr_scheduler.step()

                losses.append(epoch_loss)
                dices.append(epoch_dice)
                print(f'EPOCH: {epoch}, Loss: {epoch_loss}')
                print(f'EPOCH: {epoch}, Dice: {epoch_dice}')

                    # save train dice best checkpoint
                if epoch_dice > best_dice:
                    best_dice = epoch_dice

                plot_result(losses, 'Dice + Cross Entropy Loss', 'Loss', 'loss')
                plot_result(dices, 'Dice', 'Dice', 'dice')

def main_worker(rank, args):
    setup(rank, args.world_size)

    torch.cuda.set_device(rank)
    args.num_workers = int(args.num_workers / args.ngpus_per_node)
    args.device = torch.device(f"cuda:{rank}")
    args.rank = rank

    init_seeds(2023 + rank)

    cur_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO if rank in [-1, 0] else logging.WARN,
        filemode='w',
        filename=os.path.join(args.MODEL_SAVE_PATH, f'output_{cur_time}.log'))
    
    dataloaders = get_dataloaders(args)
    model, prompt_model = build_model(args)
    trainer = BaseTrainer(model, prompt_model, dataloaders, args)
    trainer.train()
    cleanup()


def setup(rank, world_size):
    # initialize the process group
    dist.init_process_group(
        backend='nccl',
        init_method=f'tcp://127.0.0.1:{args.port}',
        world_size=world_size,
        rank=rank
    )

def cleanup():
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
