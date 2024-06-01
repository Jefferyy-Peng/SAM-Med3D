import numpy as np
import random
import datetime
import logging
import matplotlib.pyplot as plt
import os

from models.model_single import ModelEmb
from models.unet import UNet
from utils.plot import plot_segmentation2D

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

parser = argparse.ArgumentParser()
parser.add_argument('--task_name', type=str, default='sam3d')
parser.add_argument('--click_type', type=str, default='random')
parser.add_argument('--multi_click', action='store_true', default=False)
parser.add_argument('--model_type', type=str, default='vit_b_ori')
parser.add_argument('--checkpoint', type=str, default='./ckpt/sam_med3d_turbo.pth')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--work_dir', type=str, default='work_dir')

# train
parser.add_argument('--num_workers', type=int, default=12)
parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0,1,2])
parser.add_argument('--multi_gpu', action='store_true', default=False)
parser.add_argument('--resume', action='store_true', default=False)
parser.add_argument('--allow_partial_weight', action='store_true', default=False)
parser.add_argument('--hack_prompt', type=bool, default=True)

# lr_scheduler
parser.add_argument('--lr_scheduler', type=str, default='multisteplr')
parser.add_argument('--step_size', type=list, default=[120, 180])
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--num_epochs', type=int, default=200)
parser.add_argument('--img_size', type=int, default=128)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--accumulation_steps', type=int, default=20)
parser.add_argument('--lr', type=float, default=8e-4)
parser.add_argument('--weight_decay', type=float, default=0.1)
parser.add_argument('--port', type=int, default=12361)

# prompt generator
parser.add_argument('-depth_wise', '--depth_wise', default=False, help='image size', required=False)
parser.add_argument('-order', '--order', default=85, help='image size', required=False)

args = parser.parse_args()
# args.lr = 8e-5
args.num_epochs = 1
args.accumulation_steps = 1
args.hack_prompt = True
args.multi_gpu = False

data_path = './sam3d_train/medical_preprocessed/PCa_lesion/picai/imagesTs'
label_path = './sam3d_train/medical_preprocessed/PCa_lesion/picai/labelsTs'

MODEL_SAVE_PATH = join('./work_dir', 'picai_dice_ce_beta_0.9_lr_0.001_weight_decay_0.01')

sam_model_name = 'sam_model_dice_best.pth'
prompt_model_name = 'prompt_model_dice_best.pth'

device = 'cuda:0'

model_type = 'vit_b_ori'

sam_model = sam_model_registry3D[model_type](checkpoint=join(MODEL_SAVE_PATH, sam_model_name)).to(device)
sam_model.eval()
prompt_model = ModelEmb(args=args)
prompt_model_ckpt = join(MODEL_SAVE_PATH, prompt_model_name)
with open(prompt_model_ckpt, "rb") as f:
    state_dict = torch.load(f)
prompt_model.load_state_dict(state_dict['model_state_dict'])
prompt_model.eval().to(device)

result_path = join(MODEL_SAVE_PATH, 'eval')
os.makedirs(result_path, exist_ok=True)


def get_dice_score(prev_masks, gt3D):
    def compute_dice(mask_pred, mask_gt):
        mask_threshold = 0.5

        mask_pred = (mask_pred > mask_threshold)
        mask_gt = (mask_gt > 0)

        volume_sum = mask_gt.sum() + mask_pred.sum()
        if volume_sum == 0:
            return np.NaN
        volume_intersect = (mask_gt & mask_pred).sum()
        return ((2 * volume_intersect) + 1e-5) / (volume_sum + 1e-5)

    pred_masks = (prev_masks > 0.5)
    true_masks = (gt3D > 0)
    dice_list = []
    for i in range(true_masks.shape[0]):
        dice_list.append(compute_dice(pred_masks[i], true_masks[i]))
    return (sum(dice_list) / len(dice_list)).item()


def get_test_dataloaders(args):
    test_dataset = Dataset_Union_ALL(paths=img_datas, mode='test', data_type='Ts', transform=tio.Compose([
        tio.ToCanonical(),
        # tio.CropOrPad(mask_name='label', target_shape=(args.img_size,args.img_size,args.img_size)), # crop only object region
        tio.Resize(target_shape=(args.img_size, args.img_size, args.img_size)),
        # tio.RandomFlip(axes=(0, 1, 2)),
    ]),
    threshold=1000)

    if args.multi_gpu:
        test_sampler = DistributedSampler(test_dataset)
        shuffle = False
    else:
        test_sampler = None
        shuffle = True

    test_dataloader = Union_Dataloader(
        dataset=test_dataset,
        sampler=test_sampler,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return test_dataloader

def auto_decode(sam_model, image_embedding, dense_embeddings, gt3D):
    def batch_auto_forward(sam_model, image_embedding, dense_embeddings, gt3D):

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
    low_res_masks, prev_masks = batch_auto_forward(sam_model, image_embedding, dense_embeddings, gt3D)
    return prev_masks

test_dataloader = get_test_dataloaders(args)

if not args.multi_gpu or (args.multi_gpu and args.rank == 0):
    tbar = tqdm(test_dataloader)
else:
    tbar = test_dataloader

norm_transform = tio.ZNormalization(masking_method=lambda x: x > 0)
for step, (image3D, gt3D, image_path, orig_shape) in enumerate(tbar):
    image3D = norm_transform(image3D.squeeze(dim=1))  # (N, C, W, H, D)
    image3D = image3D.unsqueeze(dim=1)
    orig_shape = tuple([i.item() for i in orig_shape])

    # get original image
    reverse_transform = tio.Compose([
        # tio.CropOrPad(mask_name='label', target_shape=orig_shape), # crop only object region
        tio.Resize(target_shape=orig_shape),
        # tio.RandomFlip(axes=(0, 1, 2)),
    ])
    subject = tio.Subject(
        image=tio.ScalarImage(tensor=image3D.squeeze(0)),
        label=tio.LabelMap(tensor=gt3D.squeeze(0)),
    )

    orig_subject = reverse_transform(subject)
    orig_img3D = orig_subject.image.data
    orig_gt3D = orig_subject.label.data

    image3D = image3D.to(device)
    gt3D = gt3D.to(device).type(torch.long)
    with amp.autocast():
        if args.hack_prompt:
            image_embedding = sam_model.image_encoder(image3D)
            dense_embeddings = prompt_model(image3D)
            prev_masks = auto_decode(sam_model, image_embedding, dense_embeddings, gt3D)
        else:
            image_embedding = sam_model.image_encoder(image3D)
            raise NotImplementedError
    # get original sized prediction
    pred_subject = tio.Subject(
        image=tio.ScalarImage(tensor=prev_masks.float().squeeze(0).detach().cpu()),
        label=tio.LabelMap(tensor=gt3D.squeeze(0).detach().cpu()),
    )
    pred_subject = reverse_transform(pred_subject)
    orig_pred_mask = pred_subject.image.data.unsqueeze(0)

    # pred_mask = prev_masks.clone().detach()
    # image_dice = get_dice_score(torch.sigmoid(pred_mask), gt3D)
    # pred_mask_b = (torch.sigmoid(pred_mask) > 0.5)
    pred_mask = orig_pred_mask.clone().detach()
    image_dice = get_dice_score(torch.sigmoid(pred_mask), orig_gt3D)
    print(image_dice)
    pred_mask_b = (torch.sigmoid(pred_mask) > 0.5)
    # plot_segmentation2D(orig_img3D.squeeze(0).squeeze(0).cpu().numpy(), pred_mask_b.squeeze(0).squeeze(0).cpu().numpy(), orig_gt3D.squeeze(0).squeeze(0).cpu().numpy(), join(result_path, str(step)+'_'+ image_path[0].split('/')[3] + '_' + image_path[0].split('/')[4]), image_dice=image_dice)
