img_datas = ['./sam3d_train/medical_preprocessed/PCa_lesion/PROMISE12_mr_unknown']

all_classes = ['PCa_lesion']

all_datasets = ['PROMISE12_mr_unknown']

# img_datas = [
# 'sam3d_train/medical_data_all/COVID_lesion/COVID1920_ct',
# 'sam3d_train/medical_data_all/COVID_lesion/Chest_CT_Scans_with_COVID-19_ct',
# 'sam3d_train/medical_data_all/adrenal/WORD_ct',
# 'sam3d_train/medical_data_all/adrenal_gland_left/AMOS2022_ct',
# 'sam3d_train/medical_data_all/adrenal_gland_left/AMOS2022_mr_unknown',
# 'sam3d_train/medical_data_all/adrenal_gland_left/BTCV_Abdomen_ct',
# 'sam3d_train/medical_data_all/adrenal_gland_left/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/adrenal_gland_right/AMOS2022_ct',
# 'sam3d_train/medical_data_all/adrenal_gland_right/AMOS2022_mr_unknown',
# 'sam3d_train/medical_data_all/adrenal_gland_right/BTCV_Abdomen_ct',
# 'sam3d_train/medical_data_all/adrenal_gland_right/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/airway/ATM2022_ct',
# 'sam3d_train/medical_data_all/aorta/AMOS2022_ct',
# 'sam3d_train/medical_data_all/aorta/AMOS2022_mr_unknown',
# 'sam3d_train/medical_data_all/aorta/BTCV_Abdomen_ct',
# 'sam3d_train/medical_data_all/aorta/SegThor_ct',
# 'sam3d_train/medical_data_all/aorta/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/autochthon_left/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/autochthon_right/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/bladder/AMOS2022_ct',
# 'sam3d_train/medical_data_all/bladder/AMOS2022_mr_unknown',
# 'sam3d_train/medical_data_all/bladder/BTCV_Cervix_ct',
# 'sam3d_train/medical_data_all/bladder/CTORG_ct',
# 'sam3d_train/medical_data_all/bladder/WORD_ct',
# 'sam3d_train/medical_data_all/bone/CTORG_ct',
# 'sam3d_train/medical_data_all/brain/CTORG_ct',
# 'sam3d_train/medical_data_all/brain/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/brain_lesion/ATLAS2_mr_t1w',
# 'sam3d_train/medical_data_all/brainstem/FeTA2022_mr_t2w',
# 'sam3d_train/medical_data_all/brainstem/HeadandNeckAutoSegmentationChallenge_ct',
# 'sam3d_train/medical_data_all/brainstem/StructSeg2019_subtask1_ct',
# 'sam3d_train/medical_data_all/buckle_rib_fracture/RibFrac2020_ct',
# 'sam3d_train/medical_data_all/caudate_left/CAUSE07_mr_unknown',
# 'sam3d_train/medical_data_all/caudate_right/CAUSE07_mr_unknown',
# 'sam3d_train/medical_data_all/cerebellum/FeTA2022_mr_t2w',
# 'sam3d_train/medical_data_all/cerebral_microbleed/VALDO_Task2_mr_t2s',
# 'sam3d_train/medical_data_all/cerebrospinal_fluid/MRBrain18_mr_t1',
# 'sam3d_train/medical_data_all/cerebrospinal_fluid/MRBrain18_mr_t1ir',
# 'sam3d_train/medical_data_all/cerebrospinal_fluid/MRBrain18_mr_t2flair',
# 'sam3d_train/medical_data_all/cerebrospinal_fluid/MRBrainS13_mr_t1',
# 'sam3d_train/medical_data_all/cerebrospinal_fluid/MRBrainS13_mr_t1ir',
# 'sam3d_train/medical_data_all/cerebrospinal_fluid/MRBrainS13_mr_t2flair',
# 'sam3d_train/medical_data_all/cerebrospinal_fluid/cSeg-2022_mr_unknown',
# 'sam3d_train/medical_data_all/cerebrospinal_fluid/iSeg2017_mr_t1',
# 'sam3d_train/medical_data_all/cerebrospinal_fluid/iSeg2017_mr_t2',
# 'sam3d_train/medical_data_all/cerebrospinal_fluid/iseg2019_mr_t1',
# 'sam3d_train/medical_data_all/cerebrospinal_fluid/iseg2019_mr_t2',
# 'sam3d_train/medical_data_all/clavicula_left/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/clavicula_right/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/colon/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/colon/WORD_ct',
# 'sam3d_train/medical_data_all/colon_cancer_primaries/MSD10_Colon_ct',
# 'sam3d_train/medical_data_all/deep_gray_matter/FeTA2022_mr_t2w',
# 'sam3d_train/medical_data_all/displaced_rib_fracture/RibFrac2020_ct',
# 'sam3d_train/medical_data_all/duodenum/AMOS2022_ct',
# 'sam3d_train/medical_data_all/duodenum/AMOS2022_mr_unknown',
# 'sam3d_train/medical_data_all/duodenum/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/duodenum/WORD_ct',
# 'sam3d_train/medical_data_all/edema/BraTS2021_mr_flair',
# 'sam3d_train/medical_data_all/edema/BraTS2021_mr_t1',
# 'sam3d_train/medical_data_all/edema/BraTS2021_mr_t1ce',
# 'sam3d_train/medical_data_all/edema/BraTS2021_mr_t2',
# 'sam3d_train/medical_data_all/edema/BrainTumour_mr_flair',
# 'sam3d_train/medical_data_all/edema/BrainTumour_mr_t1gd',
# 'sam3d_train/medical_data_all/edema/BrainTumour_mr_t1w',
# 'sam3d_train/medical_data_all/edema/BrainTumour_mr_t2w',
# 'sam3d_train/medical_data_all/enhancing_tumor/BraTS2021_mr_flair',
# 'sam3d_train/medical_data_all/enhancing_tumor/BraTS2021_mr_t1',
# 'sam3d_train/medical_data_all/enhancing_tumor/BraTS2021_mr_t1ce',
# 'sam3d_train/medical_data_all/enhancing_tumor/BraTS2021_mr_t2',
# 'sam3d_train/medical_data_all/enhancing_tumor/BrainTumour_mr_flair',
# 'sam3d_train/medical_data_all/enhancing_tumor/BrainTumour_mr_t1gd',
# 'sam3d_train/medical_data_all/enhancing_tumor/BrainTumour_mr_t1w',
# 'sam3d_train/medical_data_all/enhancing_tumor/BrainTumour_mr_t2w',
# 'sam3d_train/medical_data_all/esophagus/AMOS2022_ct',
# 'sam3d_train/medical_data_all/esophagus/AMOS2022_mr_unknown',
# 'sam3d_train/medical_data_all/esophagus/BTCV_Abdomen_ct',
# 'sam3d_train/medical_data_all/esophagus/SegThor_ct',
# 'sam3d_train/medical_data_all/esophagus/StructSeg2019_subtask2_ct',
# 'sam3d_train/medical_data_all/esophagus/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/esophagus/WORD_ct',
# 'sam3d_train/medical_data_all/external_cerebrospinal_fluid/FeTA2022_mr_t2w',
# 'sam3d_train/medical_data_all/face/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/femur_left/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/femur_right/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/gallbladder/AMOS2022_ct',
# 'sam3d_train/medical_data_all/gallbladder/AMOS2022_mr_unknown',
# 'sam3d_train/medical_data_all/gallbladder/BTCV_Abdomen_ct',
# 'sam3d_train/medical_data_all/gallbladder/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/gallbladder/WORD_ct',
# 'sam3d_train/medical_data_all/gluteus_maximus_left/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/gluteus_maximus_right/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/gluteus_medius_left/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/gluteus_medius_right/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/gluteus_minimus_left/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/gluteus_minimus_right/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/gray_matter/FeTA2022_mr_t2w',
# 'sam3d_train/medical_data_all/gray_matter/MRBrain18_mr_t1',
# 'sam3d_train/medical_data_all/gray_matter/MRBrain18_mr_t1ir',
# 'sam3d_train/medical_data_all/gray_matter/MRBrain18_mr_t2flair',
# 'sam3d_train/medical_data_all/gray_matter/MRBrainS13_mr_t1',
# 'sam3d_train/medical_data_all/gray_matter/MRBrainS13_mr_t1ir',
# 'sam3d_train/medical_data_all/gray_matter/MRBrainS13_mr_t2flair',
# 'sam3d_train/medical_data_all/gray_matter/cSeg-2022_mr_unknown',
# 'sam3d_train/medical_data_all/gray_matter/iSeg2017_mr_t1',
# 'sam3d_train/medical_data_all/gray_matter/iSeg2017_mr_t2',
# 'sam3d_train/medical_data_all/gray_matter/iseg2019_mr_t1',
# 'sam3d_train/medical_data_all/gray_matter/iseg2019_mr_t2',
# 'sam3d_train/medical_data_all/head_of_femur_left/WORD_ct',
# 'sam3d_train/medical_data_all/head_of_femur_right/WORD_ct',
# 'sam3d_train/medical_data_all/heart/SegThor_ct',
# 'sam3d_train/medical_data_all/heart/StructSeg2019_subtask2_ct',
# 'sam3d_train/medical_data_all/heart_ascending_aorta/MMWHS_ct',
# 'sam3d_train/medical_data_all/heart_atrium_left/HeartSegMRI_mr_unknown',
# 'sam3d_train/medical_data_all/heart_atrium_left/LAScarQS22Task1_mr_lge',
# 'sam3d_train/medical_data_all/heart_atrium_left/LAScarQS22Task2_mr_lge',
# 'sam3d_train/medical_data_all/heart_atrium_left/MSD02_Heart_mr_unknown',
# 'sam3d_train/medical_data_all/heart_atrium_left/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/heart_atrium_left_scars/LAScarQS22Task1_mr_lge',
# 'sam3d_train/medical_data_all/heart_atrium_right/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/heart_left_atrium_blood_cavity/MMWHS_ct',
# 'sam3d_train/medical_data_all/heart_left_ventricle_blood_cavity/MMWHS_ct',
# 'sam3d_train/medical_data_all/heart_left_ventricular_myocardium/MMWHS_ct',
# 'sam3d_train/medical_data_all/heart_myocardium/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/heart_right_atrium_blood_cavity/MMWHS_ct',
# 'sam3d_train/medical_data_all/heart_right_ventricle_blood_cavity/MMWHS_ct',
# 'sam3d_train/medical_data_all/heart_ventricle_left/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/heart_ventricle_right/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/hepatic_tumor/MSD08_HepaticVessel_ct',
# 'sam3d_train/medical_data_all/hepatic_vessels/MSD08_HepaticVessel_ct',
# 'sam3d_train/medical_data_all/hip_left/CTPelvic1k_ct',
# 'sam3d_train/medical_data_all/hip_left/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/hip_right/CTPelvic1k_ct',
# 'sam3d_train/medical_data_all/hip_right/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/hippocampus_anterior/MSD04_Hippocampus_mr_unknown',
# 'sam3d_train/medical_data_all/hippocampus_posterior/MSD04_Hippocampus_mr_unknown',
# 'sam3d_train/medical_data_all/humerus_left/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/humerus_right/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/iliac_artery_left/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/iliac_artery_right/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/iliac_vena_left/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/iliac_vena_right/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/iliopsoas_left/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/iliopsoas_right/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/inferior_vena_cava/AMOS2022_ct',
# 'sam3d_train/medical_data_all/inferior_vena_cava/AMOS2022_mr_unknown',
# 'sam3d_train/medical_data_all/inferior_vena_cava/BTCV_Abdomen_ct',
# 'sam3d_train/medical_data_all/inferior_vena_cava/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/intestine/WORD_ct',
# 'sam3d_train/medical_data_all/ischemic_stroke_lesion/ISLES2022_mr_adc',
# 'sam3d_train/medical_data_all/ischemic_stroke_lesion/ISLES2022_mr_dwi',
# 'sam3d_train/medical_data_all/kidney/AbdomenCT1K_ct',
# 'sam3d_train/medical_data_all/kidney/CTORG_ct',
# 'sam3d_train/medical_data_all/kidney/FLARE21_ct',
# 'sam3d_train/medical_data_all/kidney/KiPA22_ct',
# 'sam3d_train/medical_data_all/kidney_left/AMOS2022_ct',
# 'sam3d_train/medical_data_all/kidney_left/AMOS2022_mr_unknown',
# 'sam3d_train/medical_data_all/kidney_left/BTCV_Abdomen_ct',
# 'sam3d_train/medical_data_all/kidney_left/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/kidney_left/WORD_ct',
# 'sam3d_train/medical_data_all/kidney_right/AMOS2022_ct',
# 'sam3d_train/medical_data_all/kidney_right/AMOS2022_mr_unknown',
# 'sam3d_train/medical_data_all/kidney_right/BTCV_Abdomen_ct',
# 'sam3d_train/medical_data_all/kidney_right/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/kidney_right/WORD_ct',
# 'sam3d_train/medical_data_all/kidney_tumor/KiPA22_ct',
# 'sam3d_train/medical_data_all/left_eye/StructSeg2019_subtask1_ct',
# 'sam3d_train/medical_data_all/left_inner_ear/StructSeg2019_subtask1_ct',
# 'sam3d_train/medical_data_all/left_lens/StructSeg2019_subtask1_ct',
# 'sam3d_train/medical_data_all/left_mandible/StructSeg2019_subtask1_ct',
# 'sam3d_train/medical_data_all/left_middle_ear/StructSeg2019_subtask1_ct',
# 'sam3d_train/medical_data_all/left_optical_nerve/StructSeg2019_subtask1_ct',
# 'sam3d_train/medical_data_all/left_parotid_gland/StructSeg2019_subtask1_ct',
# 'sam3d_train/medical_data_all/left_temporal_lobes/StructSeg2019_subtask1_ct',
# 'sam3d_train/medical_data_all/left_temporomandibular_joint/StructSeg2019_subtask1_ct',
# 'sam3d_train/medical_data_all/liver/AMOS2022_ct',
# 'sam3d_train/medical_data_all/liver/AMOS2022_mr_unknown',
# 'sam3d_train/medical_data_all/liver/AbdomenCT1K_ct',
# 'sam3d_train/medical_data_all/liver/BTCV_Abdomen_ct',
# 'sam3d_train/medical_data_all/liver/CTORG_ct',
# 'sam3d_train/medical_data_all/liver/FLARE21_ct',
# 'sam3d_train/medical_data_all/liver/LITS_ct',
# 'sam3d_train/medical_data_all/liver/SLIVER07_ct',
# 'sam3d_train/medical_data_all/liver/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/liver_tumor/LITS_ct',
# 'sam3d_train/medical_data_all/lumbar_vertebra/CTPelvic1k_ct',
# 'sam3d_train/medical_data_all/lung/CTORG_ct',
# 'sam3d_train/medical_data_all/lung/LUNA16_ct',
# 'sam3d_train/medical_data_all/lung_cancer/MSD06_Lung_ct',
# 'sam3d_train/medical_data_all/lung_cancer/StructSeg2019_subtask3_ct',
# 'sam3d_train/medical_data_all/lung_infections/COVID19CTscans_ct',
# 'sam3d_train/medical_data_all/lung_left/COVID19CTscans_ct',
# 'sam3d_train/medical_data_all/lung_left/StructSeg2019_subtask2_ct',
# 'sam3d_train/medical_data_all/lung_lower_lobe_left/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/lung_lower_lobe_right/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/lung_middle_lobe_right/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/lung_node/LNDb_ct',
# 'sam3d_train/medical_data_all/lung_right/COVID19CTscans_ct',
# 'sam3d_train/medical_data_all/lung_right/StructSeg2019_subtask2_ct',
# 'sam3d_train/medical_data_all/lung_upper_lobe_left/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/lung_upper_lobe_right/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/lung_vessel/VESSEL2012_ct',
# 'sam3d_train/medical_data_all/matter_tracts/BrainPTM2021_mr_t1',
# 'sam3d_train/medical_data_all/multiple_sclerosis_lesion/LongitudinalMultipleSclerosisLesionSegmentation_mr_flair',
# 'sam3d_train/medical_data_all/multiple_sclerosis_lesion/LongitudinalMultipleSclerosisLesionSegmentation_mr_mprage',
# 'sam3d_train/medical_data_all/multiple_sclerosis_lesion/LongitudinalMultipleSclerosisLesionSegmentation_mr_pd',
# 'sam3d_train/medical_data_all/multiple_sclerosis_lesion/LongitudinalMultipleSclerosisLesionSegmentation_mr_t2',
# 'sam3d_train/medical_data_all/multiple_sclerosis_lesion/MESSEG_mr_flair',
# 'sam3d_train/medical_data_all/multiple_sclerosis_lesion/MSseg08_mr_flair',
# 'sam3d_train/medical_data_all/multiple_sclerosis_lesion/MSseg08_mr_t1',
# 'sam3d_train/medical_data_all/multiple_sclerosis_lesion/MSseg08_mr_t2',
# 'sam3d_train/medical_data_all/nasopharynx_cancer/StructSeg2019_subtask4_ct',
# 'sam3d_train/medical_data_all/non_displaced_rib_fracture/RibFrac2020_ct',
# 'sam3d_train/medical_data_all/non_enhancing_tumor/BraTS2021_mr_flair',
# 'sam3d_train/medical_data_all/non_enhancing_tumor/BraTS2021_mr_t1',
# 'sam3d_train/medical_data_all/non_enhancing_tumor/BraTS2021_mr_t1ce',
# 'sam3d_train/medical_data_all/non_enhancing_tumor/BraTS2021_mr_t2',
# 'sam3d_train/medical_data_all/non_enhancing_tumor/BrainTumour_mr_flair',
# 'sam3d_train/medical_data_all/non_enhancing_tumor/BrainTumour_mr_t1gd',
# 'sam3d_train/medical_data_all/non_enhancing_tumor/BrainTumour_mr_t1w',
# 'sam3d_train/medical_data_all/non_enhancing_tumor/BrainTumour_mr_t2w',
# 'sam3d_train/medical_data_all/optic_chiasm/StructSeg2019_subtask1_ct',
# 'sam3d_train/medical_data_all/other_pathology/WMH_mr_flair',
# 'sam3d_train/medical_data_all/other_pathology/WMH_mr_t1',
# 'sam3d_train/medical_data_all/pancreas/AMOS2022_ct',
# 'sam3d_train/medical_data_all/pancreas/AMOS2022_mr_unknown',
# 'sam3d_train/medical_data_all/pancreas/AbdomenCT1K_ct',
# 'sam3d_train/medical_data_all/pancreas/BTCV_Abdomen_ct',
# 'sam3d_train/medical_data_all/pancreas/FLARE21_ct',
# 'sam3d_train/medical_data_all/pancreas/MSD07_Pancreas_ct',
# 'sam3d_train/medical_data_all/pancreas/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/pancreas/WORD_ct',
# 'sam3d_train/medical_data_all/pancreatic_tumor_mass/MSD07_Pancreas_ct',
# 'sam3d_train/medical_data_all/pituitary/StructSeg2019_subtask1_ct',
# 'sam3d_train/medical_data_all/portal_vein_and_splenic_vein/BTCV_Abdomen_ct',
# 'sam3d_train/medical_data_all/portal_vein_and_splenic_vein/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/prostate/PROMISE12_mr_unknown',
# 'sam3d_train/medical_data_all/prostate/Prostate_MRI_Segmentation_Dataset_mr_t2w',
# 'sam3d_train/medical_data_all/prostate_and_uterus/AMOS2022_ct',
# 'sam3d_train/medical_data_all/prostate_and_uterus/AMOS2022_mr_unknown',
# 'sam3d_train/medical_data_all/prostate_peripheral_zone/MSD05_Prostate_mr_adc',
# 'sam3d_train/medical_data_all/prostate_peripheral_zone/MSD05_Prostate_mr_t2',
# 'sam3d_train/medical_data_all/prostate_transition_zone/MSD05_Prostate_mr_adc',
# 'sam3d_train/medical_data_all/prostate_transition_zone/MSD05_Prostate_mr_t2',
# 'sam3d_train/medical_data_all/pulmonary_artery/MMWHS_ct',
# 'sam3d_train/medical_data_all/pulmonary_artery/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/rectum/BTCV_Cervix_ct',
# 'sam3d_train/medical_data_all/rectum/WORD_ct',
# 'sam3d_train/medical_data_all/renal_artery/KiPA22_ct',
# 'sam3d_train/medical_data_all/renal_vein/KiPA22_ct',
# 'sam3d_train/medical_data_all/rib_left_1/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/rib_left_10/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/rib_left_11/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/rib_left_12/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/rib_left_2/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/rib_left_3/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/rib_left_4/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/rib_left_5/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/rib_left_6/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/rib_left_7/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/rib_left_8/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/rib_left_9/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/rib_right_1/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/rib_right_10/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/rib_right_11/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/rib_right_12/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/rib_right_2/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/rib_right_3/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/rib_right_4/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/rib_right_5/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/rib_right_6/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/rib_right_7/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/rib_right_8/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/rib_right_9/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/right_eye/StructSeg2019_subtask1_ct',
# 'sam3d_train/medical_data_all/right_inner_ear/StructSeg2019_subtask1_ct',
# 'sam3d_train/medical_data_all/right_lens/StructSeg2019_subtask1_ct',
# 'sam3d_train/medical_data_all/right_mandible/StructSeg2019_subtask1_ct',
# 'sam3d_train/medical_data_all/right_middle_ear/StructSeg2019_subtask1_ct',
# 'sam3d_train/medical_data_all/right_optical_nerve/StructSeg2019_subtask1_ct',
# 'sam3d_train/medical_data_all/right_parotid_gland/StructSeg2019_subtask1_ct',
# 'sam3d_train/medical_data_all/right_temporal_lobes/StructSeg2019_subtask1_ct',
# 'sam3d_train/medical_data_all/right_temporomandibular_joint/StructSeg2019_subtask1_ct',
# 'sam3d_train/medical_data_all/sacrum/CTPelvic1k_ct',
# 'sam3d_train/medical_data_all/sacrum/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/scapula_left/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/scapula_right/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/segmental_rib_fracture/RibFrac2020_ct',
# 'sam3d_train/medical_data_all/small_bowel/BTCV_Cervix_ct',
# 'sam3d_train/medical_data_all/small_bowel/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/spinal_cord/StructSeg2019_subtask1_ct',
# 'sam3d_train/medical_data_all/spinal_cord/StructSeg2019_subtask2_ct',
# 'sam3d_train/medical_data_all/spleen/AMOS2022_ct',
# 'sam3d_train/medical_data_all/spleen/AMOS2022_mr_unknown',
# 'sam3d_train/medical_data_all/spleen/AbdomenCT1K_ct',
# 'sam3d_train/medical_data_all/spleen/BTCV_Abdomen_ct',
# 'sam3d_train/medical_data_all/spleen/FLARE21_ct',
# 'sam3d_train/medical_data_all/spleen/MSD09_Spleen_ct',
# 'sam3d_train/medical_data_all/spleen/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/spleen/WORD_ct',
# 'sam3d_train/medical_data_all/stomach/AMOS2022_ct',
# 'sam3d_train/medical_data_all/stomach/AMOS2022_mr_unknown',
# 'sam3d_train/medical_data_all/stomach/BTCV_Abdomen_ct',
# 'sam3d_train/medical_data_all/stomach/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/stomach/WORD_ct',
# 'sam3d_train/medical_data_all/trachea/SegThor_ct',
# 'sam3d_train/medical_data_all/trachea/StructSeg2019_subtask2_ct',
# 'sam3d_train/medical_data_all/trachea/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/unidentified_rib_fracture/RibFrac2020_ct',
# 'sam3d_train/medical_data_all/urinary_bladder/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/uterus/BTCV_Cervix_ct',
# 'sam3d_train/medical_data_all/ventricles/FeTA2022_mr_t2w',
# 'sam3d_train/medical_data_all/vertebrae_C1/CTSpine1K_ct',
# 'sam3d_train/medical_data_all/vertebrae_C1/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/vertebrae_C2/CTSpine1K_ct',
# 'sam3d_train/medical_data_all/vertebrae_C2/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/vertebrae_C3/CTSpine1K_ct',
# 'sam3d_train/medical_data_all/vertebrae_C3/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/vertebrae_C4/CTSpine1K_ct',
# 'sam3d_train/medical_data_all/vertebrae_C4/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/vertebrae_C5/CTSpine1K_ct',
# 'sam3d_train/medical_data_all/vertebrae_C5/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/vertebrae_C6/CTSpine1K_ct',
# 'sam3d_train/medical_data_all/vertebrae_C6/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/vertebrae_C7/CTSpine1K_ct',
# 'sam3d_train/medical_data_all/vertebrae_C7/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/vertebrae_L1/CTSpine1K_ct',
# 'sam3d_train/medical_data_all/vertebrae_L1/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/vertebrae_L2/CTSpine1K_ct',
# 'sam3d_train/medical_data_all/vertebrae_L2/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/vertebrae_L3/CTSpine1K_ct',
# 'sam3d_train/medical_data_all/vertebrae_L3/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/vertebrae_L4/CTSpine1K_ct',
# 'sam3d_train/medical_data_all/vertebrae_L4/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/vertebrae_L5/CTSpine1K_ct',
# 'sam3d_train/medical_data_all/vertebrae_L5/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/vertebrae_L6/CTSpine1K_ct',
# 'sam3d_train/medical_data_all/vertebrae_T1/CTSpine1K_ct',
# 'sam3d_train/medical_data_all/vertebrae_T1/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/vertebrae_T10/CTSpine1K_ct',
# 'sam3d_train/medical_data_all/vertebrae_T10/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/vertebrae_T11/CTSpine1K_ct',
# 'sam3d_train/medical_data_all/vertebrae_T11/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/vertebrae_T12/CTSpine1K_ct',
# 'sam3d_train/medical_data_all/vertebrae_T12/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/vertebrae_T2/CTSpine1K_ct',
# 'sam3d_train/medical_data_all/vertebrae_T2/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/vertebrae_T3/CTSpine1K_ct',
# 'sam3d_train/medical_data_all/vertebrae_T3/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/vertebrae_T4/CTSpine1K_ct',
# 'sam3d_train/medical_data_all/vertebrae_T4/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/vertebrae_T5/CTSpine1K_ct',
# 'sam3d_train/medical_data_all/vertebrae_T5/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/vertebrae_T6/CTSpine1K_ct',
# 'sam3d_train/medical_data_all/vertebrae_T6/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/vertebrae_T7/CTSpine1K_ct',
# 'sam3d_train/medical_data_all/vertebrae_T7/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/vertebrae_T8/CTSpine1K_ct',
# 'sam3d_train/medical_data_all/vertebrae_T8/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/vertebrae_T9/CTSpine1K_ct',
# 'sam3d_train/medical_data_all/vertebrae_T9/Totalsegmentator_dataset_ct',
# 'sam3d_train/medical_data_all/white_matter/FeTA2022_mr_t2w',
# 'sam3d_train/medical_data_all/white_matter/MRBrain18_mr_t1',
# 'sam3d_train/medical_data_all/white_matter/MRBrain18_mr_t1ir',
# 'sam3d_train/medical_data_all/white_matter/MRBrain18_mr_t2flair',
# 'sam3d_train/medical_data_all/white_matter/MRBrainS13_mr_t1',
# 'sam3d_train/medical_data_all/white_matter/MRBrainS13_mr_t1ir',
# 'sam3d_train/medical_data_all/white_matter/MRBrainS13_mr_t2flair',
# 'sam3d_train/medical_data_all/white_matter/cSeg-2022_mr_unknown',
# 'sam3d_train/medical_data_all/white_matter/iSeg2017_mr_t1',
# 'sam3d_train/medical_data_all/white_matter/iSeg2017_mr_t2',
# 'sam3d_train/medical_data_all/white_matter/iseg2019_mr_t1',
# 'sam3d_train/medical_data_all/white_matter/iseg2019_mr_t2',
# 'sam3d_train/medical_data_all/white_matter_hyperintensity/WMH_mr_flair',
# 'sam3d_train/medical_data_all/white_matter_hyperintensity/WMH_mr_t1',
# ]
#
# all_classes = [
# 'COVID_lesion',
# 'adrenal',
# 'adrenal_gland_left',
# 'adrenal_gland_right',
# 'airway',
# 'aorta',
# 'autochthon_left',
# 'autochthon_right',
# 'bilateral_optic_nerves',
# 'bilateral_parotid_glands',
# 'bilateral_submandibular_glands',
# 'bladder',
# 'bone',
# 'brain',
# 'brain_lesion',
# 'brainstem',
# 'buckle_rib_fracture',
# 'caudate_left',
# 'caudate_right',
# 'cerebellum',
# 'cerebral_microbleed',
# 'cerebrospinal_fluid',
# 'clavicula_left',
# 'clavicula_right',
# 'cocygis',
# 'colon',
# 'colon_cancer_primaries',
# 'deep_gray_matter',
# 'displaced_rib_fracture',
# 'duodenum',
# 'edema',
# 'enhancing_tumor',
# 'esophagus',
# 'external_cerebrospinal_fluid',
# 'face',
# 'femur_left',
# 'femur_right',
# 'gallbladder',
# 'gluteus_maximus_left',
# 'gluteus_maximus_right',
# 'gluteus_medius_left',
# 'gluteus_medius_right',
# 'gluteus_minimus_left',
# 'gluteus_minimus_right',
# 'gray_matter',
# 'head_of_femur_left',
# 'head_of_femur_right',
# 'heart',
# 'heart_ascending_aorta',
# 'heart_atrium_left',
# 'heart_atrium_left_scars',
# 'heart_atrium_right',
# 'heart_blood_pool',
# 'heart_left_atrium_blood_cavity',
# 'heart_left_ventricle_blood_cavity',
# 'heart_left_ventricular_myocardium',
# 'heart_myocardium',
# 'heart_myocardium_left',
# 'heart_right_atrium_blood_cavity',
# 'heart_right_ventricle_blood_cavity',
# 'heart_ventricle_left',
# 'heart_ventricle_right',
# 'hepatic_tumor',
# 'hepatic_vessels',
# 'hip_left',
# 'hip_right',
# 'hippocampus_anterior',
# 'hippocampus_posterior',
# 'humerus_left',
# 'humerus_right',
# 'iliac_artery_left',
# 'iliac_artery_right',
# 'iliac_vena_left',
# 'iliac_vena_right',
# 'iliopsoas_left',
# 'iliopsoas_right',
# 'inferior_vena_cava',
# 'intestine',
# 'ischemic_stroke_lesion',
# 'kidney',
# 'kidney_cyst',
# 'kidney_left',
# 'kidney_right',
# 'kidney_tumor',
# 'left_eye',
# 'left_inner_ear',
# 'left_lens',
# 'left_mandible',
# 'left_middle_ear',
# 'left_optical_nerve',
# 'left_parotid_gland',
# 'left_temporal_lobes',
# 'left_temporomandibular_joint',
# 'left_ventricular_blood_pool',
# 'left_ventricular_myocardial_edema',
# 'left_ventricular_myocardial_scars',
# 'left_ventricular_normal_myocardium',
# 'liver',
# 'liver_tumor',
# 'lumbar_vertebra',
# 'lung',
# 'lung_cancer',
# 'lung_infections',
# 'lung_left',
# 'lung_lower_lobe_left',
# 'lung_lower_lobe_right',
# 'lung_middle_lobe_right',
# 'lung_node',
# 'lung_right',
# 'lung_upper_lobe_left',
# 'lung_upper_lobe_right',
# 'lung_vessel',
# 'mandible',
# 'matter_tracts',
# 'multiple_sclerosis_lesion',
# 'myocardial_infarction',
# 'nasopharynx_cancer',
# 'no_reflow',
# 'non_displaced_rib_fracture',
# 'non_enhancing_tumor',
# 'optic_chiasm',
# 'other_pathology',
# 'pancreas',
# 'pancreatic_tumor_mass',
# 'pituitary',
# 'portal_vein_and_splenic_vein',
# 'prostate',
# 'prostate_and_uterus',
# 'prostate_peripheral_zone',
# 'prostate_transition_zone',
# 'pulmonary_artery',
# 'rectum',
# 'renal_artery',
# 'renal_vein',
# 'rib_left_1',
# 'rib_left_10',
# 'rib_left_11',
# 'rib_left_12',
# 'rib_left_2',
# 'rib_left_3',
# 'rib_left_4',
# 'rib_left_5',
# 'rib_left_6',
# 'rib_left_7',
# 'rib_left_8',
# 'rib_left_9',
# 'rib_right_1',
# 'rib_right_10',
# 'rib_right_11',
# 'rib_right_12',
# 'rib_right_2',
# 'rib_right_3',
# 'rib_right_4',
# 'rib_right_5',
# 'rib_right_6',
# 'rib_right_7',
# 'rib_right_8',
# 'rib_right_9',
# 'right_eye',
# 'right_inner_ear',
# 'right_lens',
# 'right_mandible',
# 'right_middle_ear',
# 'right_optical_nerve',
# 'right_parotid_gland',
# 'right_temporal_lobes',
# 'right_temporomandibular_joint',
# 'right_ventricular_blood_pool',
# 'sacrum',
# 'scapula_left',
# 'scapula_right',
# 'segmental_rib_fracture',
# 'small_bowel',
# 'spinal_cord',
# 'spleen',
# 'stomach',
# 'trachea',
# 'unidentified_rib_fracture',
# 'urinary_bladder',
# 'uterus',
# 'ventricles',
# 'vertebrae_C1',
# 'vertebrae_C2',
# 'vertebrae_C3',
# 'vertebrae_C4',
# 'vertebrae_C5',
# 'vertebrae_C6',
# 'vertebrae_C7',
# 'vertebrae_L1',
# 'vertebrae_L2',
# 'vertebrae_L3',
# 'vertebrae_L4',
# 'vertebrae_L5',
# 'vertebrae_L6',
# 'vertebrae_T1',
# 'vertebrae_T10',
# 'vertebrae_T11',
# 'vertebrae_T12',
# 'vertebrae_T13',
# 'vertebrae_T2',
# 'vertebrae_T3',
# 'vertebrae_T4',
# 'vertebrae_T5',
# 'vertebrae_T6',
# 'vertebrae_T7',
# 'vertebrae_T8',
# 'vertebrae_T9',
# 'white_matter',
# 'white_matter_hyperintensity',
# ]
#
# all_datasets = [
# 'AMOS2022_ct',
# 'AMOS2022_mr_unknown',
# 'ATLAS2_mr_t1w',
# 'ATM2022_ct',
# 'AbdomenCT1K_ct',
# 'BTCV_Abdomen_ct',
# 'BTCV_Cervix_ct',
# 'BraTS2021_mr_flair',
# 'BraTS2021_mr_t1',
# 'BraTS2021_mr_t1ce',
# 'BraTS2021_mr_t2',
# 'BrainPTM2021_mr_t1',
# 'BrainTumour_mr_flair',
# 'BrainTumour_mr_t1gd',
# 'BrainTumour_mr_t1w',
# 'BrainTumour_mr_t2w',
# 'CAUSE07_mr_unknown',
# 'COVID1920_ct',
# 'COVID19CTscans_ct',
# 'CTORG_ct',
# 'CTPelvic1k_ct',
# 'CTSpine1K_ct',
# 'Chest_CT_Scans_with_COVID-19_ct',
# 'FLARE21_ct',
# 'FeTA2022_mr_t2w',
# 'HeadandNeckAutoSegmentationChallenge_ct',
# 'HeartSegMRI_mr_unknown',
# 'ISLES2022_mr_adc',
# 'ISLES2022_mr_dwi',
# 'KiPA22_ct',
# 'LAScarQS22Task1_mr_lge',
# 'LAScarQS22Task2_mr_lge',
# 'LITS_ct',
# 'LNDb_ct',
# 'LUNA16_ct',
# 'LongitudinalMultipleSclerosisLesionSegmentation_mr_flair',
# 'LongitudinalMultipleSclerosisLesionSegmentation_mr_mprage',
# 'LongitudinalMultipleSclerosisLesionSegmentation_mr_pd',
# 'LongitudinalMultipleSclerosisLesionSegmentation_mr_t2',
# 'MESSEG_mr_flair',
# 'MMWHS_ct',
# 'MRBrain18_mr_t1',
# 'MRBrain18_mr_t1ir',
# 'MRBrain18_mr_t2flair',
# 'MRBrainS13_mr_t1',
# 'MRBrainS13_mr_t1ir',
# 'MRBrainS13_mr_t2flair',
# 'MSD02_Heart_mr_unknown',
# 'MSD04_Hippocampus_mr_unknown',
# 'MSD05_Prostate_mr_adc',
# 'MSD05_Prostate_mr_t2',
# 'MSD06_Lung_ct',
# 'MSD07_Pancreas_ct',
# 'MSD08_HepaticVessel_ct',
# 'MSD09_Spleen_ct',
# 'MSD10_Colon_ct',
# 'MSseg08_mr_flair',
# 'MSseg08_mr_t1',
# 'MSseg08_mr_t2',
# 'PROMISE12_mr_unknown',
# 'Prostate_MRI_Segmentation_Dataset_mr_t2w',
# 'RibFrac2020_ct',
# 'SLIVER07_ct',
# 'SegThor_ct',
# 'StructSeg2019_subtask1_ct',
# 'StructSeg2019_subtask2_ct',
# 'StructSeg2019_subtask3_ct',
# 'StructSeg2019_subtask4_ct',
# 'Totalsegmentator_dataset_ct',
# 'VALDO_Task2_mr_t2s',
# 'VESSEL2012_ct',
# 'WMH_mr_flair',
# 'WMH_mr_t1',
# 'WORD_ct',
# 'cSeg-2022_mr_unknown',
# 'iSeg2017_mr_t1',
# 'iSeg2017_mr_t2',
# 'iseg2019_mr_t1',
# 'iseg2019_mr_t2',
# 'mnms_mr_unknown',
# ]