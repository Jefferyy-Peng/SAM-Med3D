# -*- encoding: utf-8 -*-
'''
@File    :   prepare_data_from_nnUNet.py
@Time    :   2023/12/10 23:07:39
@Author  :   Haoyu Wang 
@Contact :   small_dark@sina.com
@Brief   :   pre-process nnUNet-style dataset into SAM-Med3D-style
'''

import os.path as osp
import os
import json
import shutil
import matplotlib.pyplot as plt
import nibabel as nib
from tqdm import tqdm
import torchio as tio

def resample_nii(input_path: str, output_path: str, target_spacing: tuple = (1, 1, 1), n=None, reference_image=None, mode="linear"):
    """
    Resample a nii.gz file to a specified spacing using torchio.

    Parameters:
    - input_path: Path to the input .nii.gz file.
    - output_path: Path to save the resampled .nii.gz file.
    - target_spacing: Desired spacing for resampling. Default is (1.5, 1.5, 1.5).
    """
    
    # Load the nii.gz file using torchio
    subject = tio.Subject(
        img=tio.ScalarImage(input_path)
    )
    resampler = tio.Resample(target=target_spacing, image_interpolation=mode)
    resampled_subject = resampler(subject)

    if(n!=None):
        image = resampled_subject.img
        tensor_data = image.data
        if(isinstance(n, int)):
            n = [n]
        for ni in n:
            tensor_data[tensor_data == ni] = -1
        tensor_data[tensor_data != -1] = 0
        tensor_data[tensor_data != 0] = 1
        save_image = tio.ScalarImage(tensor=tensor_data, affine=image.affine)
        reference_size = reference_image.shape[1:]  # omitting the channel dimension
        cropper_or_padder = tio.CropOrPad(reference_size)
        save_image = cropper_or_padder(save_image)
    else:
        save_image = resampled_subject.img

    
    save_image.save(output_path)

dataset_root = "/home/yunxiangpeng/PycharmProjects/SAM-Med3D/sam3d_train/medical_data_all"
dataset_list = [
    'prostate/PROMISE12_mr_unknown',
]

target_dir = "/home/yunxiangpeng/PycharmProjects/SAM-Med3D/sam3d_train/medical_preprocessed"


for dataset in dataset_list:
    dataset_dir = osp.join(dataset_root, dataset)
    meta_info = json.load(open(osp.join(dataset_dir, "dataset.json")))

    print(meta_info['name'], meta_info['modality'])
    num_classes = len(meta_info["labels"])-1
    print("num_classes:", num_classes, meta_info["labels"])
    resample_dir = osp.join(dataset_dir, "imagesTr_1.5") 
    os.makedirs(resample_dir, exist_ok=True)
    for idx, cls_name in meta_info["labels"].items():
        if cls_name == 'background':
            continue
        cls_name = cls_name.replace(" ", "_")
        idx = int(idx)
        dataset_name = dataset.split("/", maxsplit=1)[1]
        target_cls_dir = osp.join(target_dir, cls_name, dataset_name)
        target_img_dir = osp.join(target_cls_dir, "imagesTr")
        target_gt_dir = osp.join(target_cls_dir, "labelsTr")
        os.makedirs(target_img_dir, exist_ok=True)
        os.makedirs(target_gt_dir, exist_ok=True)
        for item in tqdm(meta_info["training"], desc=f"{dataset_name}-{cls_name}"):
            imgTr, gtTr = item["image"], item["label"]
            imgTr = osp.join(dataset_dir, 'imagesTr', imgTr.replace("./", ""))
            gtTr = osp.join(dataset_dir, 'labelsTr', gtTr.replace("./", ""))
            resample_img = osp.join(resample_dir, osp.basename(imgTr))
            resample_nii(imgTr, resample_img)

            target_img_path = osp.join(target_img_dir, osp.basename(resample_img))
            target_gt_path = osp.join(target_gt_dir, osp.basename(gtTr))

            gt_img = nib.load(gtTr)
            spacing = tuple(gt_img.header['pixdim'][1:4])
            spacing_voxel = spacing[0] * spacing[1] * spacing[2]
            gt_arr = gt_img.get_fdata()
            gt_arr[gt_arr != idx] = 0
            gt_arr[gt_arr != 0] = 1
            volume = gt_arr.sum()*spacing_voxel
            if(volume<10): 
                print("skip", target_img_path)
                continue

            reference_image = tio.ScalarImage(resample_img)
            if(meta_info['name']=="kits23" and idx==1):
                resample_nii(gtTr, target_gt_path, n=[1,2,3], reference_image=reference_image, mode="nearest")
            else:
                resample_nii(gtTr, target_gt_path, n=idx, reference_image=reference_image, mode="nearest")
            shutil.copy(resample_img, target_img_path)
        resample_dir = osp.join(dataset_dir, "imagesTs_1.5")
        os.makedirs(resample_dir, exist_ok=True)
        target_img_dir = osp.join(target_cls_dir, "imagesTs")
        target_gt_dir = osp.join(target_cls_dir, "labelsTs")
        os.makedirs(target_img_dir, exist_ok=True)
        os.makedirs(target_gt_dir, exist_ok=True)
        for item in tqdm(meta_info["test"], desc=f"{dataset_name}-{cls_name}"):
            imgTs, gtTs = item["image"], item["label"]
            imgTs = osp.join(dataset_dir, 'imagesTs', imgTs.replace("./", ""))
            gtTs = osp.join(dataset_dir, 'labelsTs', gtTs.replace("./", ""))
            resample_img = osp.join(resample_dir, osp.basename(imgTs))
            resample_nii(imgTs, resample_img)

            target_img_path = osp.join(target_img_dir, osp.basename(resample_img))
            target_gt_path = osp.join(target_gt_dir, osp.basename(gtTs))

            gt_img = nib.load(gtTs)
            spacing = tuple(gt_img.header['pixdim'][1:4])
            spacing_voxel = spacing[0] * spacing[1] * spacing[2]
            gt_arr = gt_img.get_fdata()
            gt_arr[gt_arr != idx] = 0
            gt_arr[gt_arr != 0] = 1
            volume = gt_arr.sum()*spacing_voxel
            if(volume<10):
                print("skip", target_img_path)
                continue

            reference_image = tio.ScalarImage(resample_img)
            if(meta_info['name']=="kits23" and idx==1):
                resample_nii(gtTs, target_gt_path, n=[1,2,3], reference_image=reference_image, mode="nearest")
            else:
                resample_nii(gtTs, target_gt_path, n=idx, reference_image=reference_image, mode="nearest")
            shutil.copy(resample_img, target_img_path)



