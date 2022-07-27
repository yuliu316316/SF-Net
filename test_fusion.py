
import sys
sys.path.append(".")
import SimpleITK as sitk
import os
import numpy as np
import nibabel as nib
import argparse
import torch
from tqdm import tqdm
from utils import BratsDataset
from config import config
from pandas import read_csv
from utils import combine_labels_predicting, dim_recovery
from SFNet import SFNet
def init_args():

    parser = argparse.ArgumentParser()
    # parser.add_argument('-a', '--attention', type=int, help='choose from 0, 1, 2', default=0)
    parser.add_argument('-g', '--num_gpu', type=int, help='Can be 0, 1, 2, 4', default=1)
    parser.add_argument('-s', '--save_folder', type=str, help='The folder of the saved model', default='saved_pth')
    parser.add_argument('-f', '--checkpoint_file', type=str, help='name of the saved pth file', default='SF-Net.pth')
    parser.add_argument('--train', type=bool, help='make prediction on training data', default=False)
    parser.add_argument('--test', type=bool, help='make prediction on testing data', default=True)
    parser.add_argument('--seglabel', type=int, help='whether to train the model with 1 or all 3 labels', default=0)
    parser.add_argument('-i', '--image_shape', type=int,  nargs='+', help='The shape of input tensor;'
                                                                    'have to be dividable by 16 (H, W, D)',
                        default=[128, 192, 160])
    parser.add_argument('-t', '--tta', type=bool, help='Whether to implement test-time augmentation;', default=False)

    return parser.parse_args()


args = init_args()
num_gpu = args.num_gpu
tta = args.tta
config["cuda_devices"] = True
if num_gpu == 0:
    config["cuda_devices"] = None
elif num_gpu == 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
elif num_gpu == 2:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
elif num_gpu == 4:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
config["batch_size"] = 1
seglabel_idx = args.seglabel
label_list = [None, "WT", "TC", "ET"]   # None represents using all 3 labels
dice_list = [None, "dice_wt", "dice_tc", "dice_et"]
seg_label = label_list[seglabel_idx]  # used for data generation
seg_dice = dice_list[seglabel_idx]  # used for dice calculation
config["image_shape"] = args.image_shape
config["checkpoint_file"] = args.checkpoint_file
config["checkpoint_path"] = os.path.join(config["base_path"], "models")
config['saved_model_path'] = os.path.join(config["checkpoint_path"], config["checkpoint_file"])
config["prediction_dir"] = os.path.join(config["base_path"], "pred_fusion", config["checkpoint_file"].split(".pth")[0])
config["load_from_data_parallel"] = True  # Load model trained on multi-gpu to predict on single gpu.
config["predict_from_train_data"] = args.train
config["predict_from_test_data"] = args.test
#config["test_path"] = os.path.join(config["base_path"], "data", "MICCAI_BraTS2020_ValidationData")
if config["predict_from_test_data"]:
    config["test_path"] = "./MICCAI_BraTS2020_ValidationData"
if config["predict_from_train_data"]:
    config["test_path"] = "./MICCAI_BraTS2020_trainingData"
config["input_shape"] = tuple([config["batch_size"]] + [config["nb_channels"]] + list(config["image_shape"]))
config["enable"] = False
config["seg_label"] = seg_label                             # used for data generation
config["num_labels"] = 1 if config["seg_label"] else 3      # used for model constructing
config["seg_dice"] = seg_dice                               # used for dice calculation

config["activation"] = "relu"
if "sin" in config["checkpoint_file"]:
    config["activation"] = "sin"

config["concat"] = False
if "cat" in config["checkpoint_file"]:
    config["concat"] = True

config["attention"] = False
if "att" in config["checkpoint_file"]:
    config["attention"] = True



def init_model_from_states(config):

    print("Init model...")
    model = SFNet(config=config)
    if config["cuda_devices"] is not None:
        if num_gpu > 0:
            model = torch.nn.DataParallel(model)   # multi-gpu inference
        model = model.cuda()
    checkpoint = torch.load(config['saved_model_path'], map_location='cpu')
    state_dict = checkpoint["state_dict"]
    if not config["load_from_data_parallel"]:
        model.load_state_dict(state_dict)
    else:
        from collections import OrderedDict     # Load state_dict from checkpoint model trained by multi-gpu
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if not "vae" in k:
                if "module." in k:
                    new_state_dict[k] = v
                # name = k[7:]
                else:
                    name = "module." + k    # fix the bug of missing keywords caused by data parallel
                    new_state_dict[name] = v

        model.load_state_dict(new_state_dict)

    return model



def predict(name_list, model):

    model.eval()
    config["test_patients"] = name_list
    config["tta_idx"] = 0   # 0 indices no test-time augmentation;
    if not os.path.exists(config["prediction_dir"]):
        os.mkdir(config["prediction_dir"])

    tmp_dir = "../tmp_result_{}".format(config["checkpoint_file"][:-4])
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    # For testing time data augment
    tta_idx_limit = 8 if tta else 1
    for tta_idx in range(tta_idx_limit):
        config["tta_idx"] = tta_idx
        if tta:
            print("starting evaluation of the {} mirror flip of Test-Time-Augmentation".format(tta_idx))
        data_set = BratsDataset(phase="test", config=config)
        valildation_loader = torch.utils.data.DataLoader(dataset=data_set,
                                                         batch_size=config["batch_size"],
                                                         shuffle=False,
                                                         pin_memory=True)
        predict_process = tqdm(valildation_loader)
        for idx, inputs in enumerate(predict_process):
            if idx > 0:
                predict_process.set_description("processing {} picture".format(idx))

            if config["cuda_devices"] is not None:
                inputs = inputs.type(torch.FloatTensor)
                inputs = inputs.cuda()
            with torch.no_grad():

                outputs = model(inputs)
            patient_filename = name_list[idx]
            print(patient_filename)
            print(outputs.shape)
            # affine = nib.load(os.path.join(config["test_path"], patient_filename, patient_filename + '_t2.nii.gz'))
            
            imgs_t2 = sitk.ReadImage(os.path.join(config["test_path"], patient_filename, patient_filename + '_t2.nii.gz'))
            imgs_flair = sitk.ReadImage(os.path.join(config["test_path"], patient_filename, patient_filename + '_flair.nii.gz'))
            imgs_t1ce = sitk.ReadImage(os.path.join(config["test_path"], patient_filename, patient_filename + '_t1ce.nii.gz'))
            
            imgs_npy1 = sitk.GetArrayFromImage(imgs_t2)
            imgs_flair_npy = sitk.GetArrayFromImage(imgs_flair)
            imgs_t1ce_npy = sitk.GetArrayFromImage(imgs_t1ce)

            imgs_npy1 = torch.tensor(imgs_npy1[2:146, 24:216, 24:216], dtype=torch.float32).cuda()
            imgs_flair_npy = torch.tensor(imgs_flair_npy[2:146, 24:216, 24:216], dtype=torch.float32).cuda()
            imgs_t1ce_npy = torch.tensor(imgs_t1ce_npy[2:146, 24:216, 24:216], dtype=torch.float32).cuda()

            outputs1 = outputs[0, 3:4, :, :, :]
            outputs2 = outputs[0, 4:5, :, :, :]
            ones1 = torch.ones([1, 144, 192, 192]).cuda()
            fusion1 = outputs1.mul(imgs_npy1) + (ones1 - outputs1).mul(imgs_t1ce_npy)
            fusion2 = outputs2.mul(imgs_flair_npy) + (ones1 - outputs2).mul(imgs_t1ce_npy)

            fusion1= np.array(fusion1.cpu())
            fusion2 = np.array(fusion2.cpu())
            # print(imgs_npy.shape)
            # output_array = np.array(outputs.cpu())  # can't convert tensor in GPU directly
            # print(output_array.shape)
            brain_mask = np.zeros((2,155,240,240),dtype=np.float32)



            # fusion1 = torch.mul(outputs1, imgs_t1ce_npy) + torch.mul((ones1 - outputs1), imgs_npy1)
            # fusion2 = torch.mul(outputs2, imgs_t1ce_npy) + torch.mul((ones1 - outputs2), imgs_flair_npy)
            brain_mask[0, 2:146, 24:216, 24:216] = fusion1
            brain_mask[1, 2:146, 24:216, 24:216] = fusion2

            saveout1 = sitk.GetImageFromArray(brain_mask[0,:,:,:])
            saveout2 = sitk.GetImageFromArray(brain_mask[1, :, :, :])

            sitk.WriteImage(saveout1, os.path.join(config["prediction_dir"], patient_filename + '.nii.gz'))
            sitk.WriteImage(saveout2, os.path.join(config["prediction_dir"], patient_filename + '1.nii.gz'))

if __name__ == "__main__":

    model = init_model_from_states(config)

    if config["predict_from_test_data"]:
        mapping_file_path = os.path.join(config["test_path"], "name_mapping_validation_data.csv")
        name_mapping = read_csv(mapping_file_path)
#        val_list = name_mapping["BraTS_2019_subject_ID"].tolist()
        val_list = name_mapping["BraTS_2020_subject_ID"].tolist()
    else:
        if config["predict_from_train_data"]:
            mapping_file_path = os.path.join(config["test_path"], "name_mapping.csv")
        else:
            mapping_file_path = os.path.join(config["test_path"], "name_mapping_validation_data.csv")
        name_mapping = read_csv(mapping_file_path)
        val_list = name_mapping["BraTS_2020_subject_ID"].tolist()
#        val_list = name_mapping["BraTS_2019_subject_ID"].tolist()
    predict(val_list, model)
