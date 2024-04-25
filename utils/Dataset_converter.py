import os
import re
import shutil

import SimpleITK as sitk
import json

convert_prostate12 = False
convert_picai = True
############################################################# convert prostate12
if convert_prostate12:
    root_path = '/home/yunxiangpeng/Datasets/MRI/picai'
    train_path = os.path.join(root_path, 'Train')
    test_path = os.path.join(root_path, 'Test')
    output_root = '/home/yunxiangpeng/PycharmProjects/SAM-Med3D/sam3d_train/medical_data_all/prostate/picai'

    json_file = os.path.join(root_path)

    task_name = 'prostate12'
    tensorImageSize = '4D'
    modality = {"0": "T2W"}
    labels = {
            "0": "background",
            "1": "PCa lesion"
        }

    # Set to store unique cases
    unique_cases = set()

    # Iterate over each file in the directory
    for filename in os.listdir(train_path):
        # Search for the pattern in the filename
        match = pattern.search(filename)
        if match:
            # Add the found case identifier to the set
            unique_cases.add(match.group())

    # Convert the set to a sorted list
    train_case_list = sorted(unique_cases)
    os.makedirs(os.path.join(output_root, 'imagesTr'), exist_ok=True)
    os.makedirs(os.path.join(output_root, 'labelsTr'), exist_ok=True)
    num_train = len(train_case_list)
    train_case_dict_list = []

    for case in train_case_list:
        image_file_path = os.path.join(train_path, case + '.mhd')
        segment_file_path = os.path.join(train_path, case + '_segmentation.mhd')
        output_train_image_filepath = os.path.join(output_root, 'imagesTr', 'PROMISE12_' + case.replace('Case', '') + '_0000.nii.gz')
        output_train_label_filepath = os.path.join(output_root, 'labelsTr', 'PROMISE12_' + case.replace('Case', '') + '_0000.nii.gz')
        train_case_dict_list.append({'image': os.path.join('.', 'PROMISE12_' + case.replace('Case', '') + '_0000.nii.gz'), 'label': os.path.join('.', 'PROMISE12_' + case.replace('Case', '') + '_0000.nii.gz')})
        image = sitk.ReadImage(image_file_path)
        segment = sitk.ReadImage(segment_file_path)
        sitk.WriteImage(image, output_train_image_filepath, True)
        sitk.WriteImage(segment, output_train_label_filepath, True)

    pattern = re.compile(r'Case\d+')

    # Set to store unique cases
    unique_cases = set()

    # Iterate over each file in the directory
    for filename in os.listdir(test_path):
        # Search for the pattern in the filename
        match = pattern.search(filename)
        if match:
            # Add the found case identifier to the set
            unique_cases.add(match.group())

    # Convert the set to a sorted list
    test_case_list = sorted(unique_cases)
    os.makedirs(os.path.join(output_root, 'imagesTs'), exist_ok=True)
    os.makedirs(os.path.join(output_root, 'labelsTs'), exist_ok=True)
    num_test = len(test_case_list)
    test_case_dict_list = []

    for case in test_case_list:
        image_file_path = os.path.join(test_path, case + '.mhd')
        segment_file_path = os.path.join(test_path, case + '_segmentation.mhd')
        output_test_image_filepath = os.path.join(output_root, 'imagesTs', 'PROMISE12_' + str(int(case.replace('Case', ''))+50) + '_0000.nii.gz')
        output_test_label_filepath = os.path.join(output_root, 'labelsTs', 'PROMISE12_' + str(int(case.replace('Case', ''))+50) + '_0000.nii.gz')
        test_case_dict_list.append({'image': os.path.join('.', 'PROMISE12_' + str(int(case.replace('Case', ''))+50) + '_0000.nii.gz'), 'label': os.path.join('.', 'PROMISE12_' + str(int(case.replace('Case', ''))+50) + '_0000.nii.gz')})
        image = sitk.ReadImage(image_file_path)
        segment = sitk.ReadImage(segment_file_path)
        sitk.WriteImage(image, output_test_image_filepath, True)
        sitk.WriteImage(segment, output_test_label_filepath, True)

    json_dict = {'name': task_name, 'modality': modality, 'labels':labels, 'numTraining': num_train, 'training': train_case_dict_list, 'numTest': num_test, 'test': test_case_dict_list}
    json_path = os.path.join(output_root, 'dataset.json')
    with open(json_path, 'w') as file:
        # Write the dictionary to file as JSON
        json.dump(json_dict, file, indent=4)

############################################################ convert picai
if convert_picai:
    root_path = '/home/yunxiangpeng/Datasets/MRI/picai'
    img_path = os.path.join(root_path, 'Train', 'images')
    output_root = '/home/yunxiangpeng/PycharmProjects/SAM-Med3D/sam3d_train/medical_data_all/prostate/picai'
    AI_lesion_annotation_path = '/home/yunxiangpeng/Datasets/MRI/picai/picai_labels/csPCa_lesion_delineations/AI/Bosma22a'

    meta_path = os.path.join(root_path, 'dataset.json')
    with open(meta_path, 'r') as file:
        # Write the dictionary to file as JSON
        meta = json.load(file)
    # get train test split
    split_path = os.path.join(root_path, 'splits.json')
    with open(split_path, 'r') as file:
        # Write the dictionary to file as JSON
        splits = json.load(file)
    use_split = splits[0]
    train_cases_list = []
    test_cases_lists = []

    os.makedirs(os.path.join(output_root, 'imagesTr'), exist_ok=True)
    os.makedirs(os.path.join(output_root, 'labelsTr'), exist_ok=True)
    os.makedirs(os.path.join(output_root, 'imagesTs'), exist_ok=True)
    os.makedirs(os.path.join(output_root, 'labelsTs'), exist_ok=True)

    for train_case in use_split['train']:
        for i in range(3):
            if os.path.exists(os.path.join(img_path, train_case + f'_000{str(i)}.nii.gz')):
                shutil.copy(os.path.join(img_path, train_case + f'_000{str(i)}.nii.gz'), os.path.join(output_root, 'imagesTr', train_case + f'_000{str(i)}.nii.gz'))
                shutil.copy(os.path.join(AI_lesion_annotation_path, train_case + '.nii.gz'), os.path.join(output_root, 'labelsTr', train_case + f'_000{str(i)}.nii.gz'))
            else:
                continue
    for test_case in use_split['val']:
        for i in range(3):
            if os.path.exists(os.path.join(img_path, test_case + f'_000{str(i)}.nii.gz')):
                shutil.copy(os.path.join(img_path), test_case + f'_000{str(i)}.nii.gz',
                            os.path.join(output_root, 'imagesTs', test_case + f'_000{str(i)}.nii.gz'))
                shutil.copy(os.path.join(AI_lesion_annotation_path, test_case + '.nii.gz'),
                            os.path.join(output_root, 'labelsTs', test_case + f'_000{str(i)}.nii.gz'))
            else:
                continue



    pattern = re.compile(r'Case\d+')

    # Set to store unique cases
    unique_cases = set()

    # Iterate over each file in the directory
    for filename in os.listdir(test_path):
        # Search for the pattern in the filename
        match = pattern.search(filename)
        if match:
            # Add the found case identifier to the set
            unique_cases.add(match.group())

    # Convert the set to a sorted list
    test_case_list = sorted(unique_cases)
    os.makedirs(os.path.join(output_root, 'imagesTs'), exist_ok=True)
    os.makedirs(os.path.join(output_root, 'labelsTs'), exist_ok=True)
    num_test = len(test_case_list)
    test_case_dict_list = []

    for case in test_case_list:
        image_file_path = os.path.join(test_path, case + '.mhd')
        segment_file_path = os.path.join(test_path, case + '_segmentation.mhd')
        output_test_image_filepath = os.path.join(output_root, 'imagesTs', 'PROMISE12_' + str(
            int(case.replace('Case', '')) + 50) + '_0000.nii.gz')
        output_test_label_filepath = os.path.join(output_root, 'labelsTs', 'PROMISE12_' + str(
            int(case.replace('Case', '')) + 50) + '_0000.nii.gz')
        test_case_dict_list.append(
            {'image': os.path.join('.', 'PROMISE12_' + str(int(case.replace('Case', '')) + 50) + '_0000.nii.gz'),
             'label': os.path.join('.', 'PROMISE12_' + str(int(case.replace('Case', '')) + 50) + '_0000.nii.gz')})
        image = sitk.ReadImage(image_file_path)
        segment = sitk.ReadImage(segment_file_path)
        sitk.WriteImage(image, output_test_image_filepath, True)
        sitk.WriteImage(segment, output_test_label_filepath, True)

    json_dict = {'name': task_name, 'modality': modality, 'labels': labels, 'numTraining': num_train,
                 'training': train_case_dict_list, 'numTest': num_test, 'test': test_case_dict_list}
    json_path = os.path.join(output_root, 'dataset.json')
    with open(json_path, 'w') as file:
        # Write the dictionary to file as JSON
        json.dump(json_dict, file, indent=4)