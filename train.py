
import os

os.environ['PYTHONHASHSEED'] = '0'
import pandas as pd
import sys
sys.path.append(".")

from SFNet import SFNet
import numpy as np
import random
import torch
import os
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from utils import Logger, load_old_model, poly_lr_scheduler,train_epoch,val_epoch

from loss import CombinedLoss, SoftDiceLoss

from utils import BratsDataset
import argparse
from config import config

# set seed
seed_num = 64
np.random.seed(seed_num)
random.seed(seed_num)

def init_args():

    parser = argparse.ArgumentParser()
    # parser.add_argument('--batch_size', type=int, help='The batch size of the test images', default=2)
    parser.add_argument('-e', '--epoch', type=str, help='The number of epochs of training', default=200)
    parser.add_argument('-g', '--num_gpu', type=int, help='Can be 0, 1, 2, 4', default=2)
    parser.add_argument('-a', '--attention', type=int, help='whether to use attention blocks, 0, 1, 2', default=0)
    parser.add_argument('--concat', type=bool, help='whether to use add & concatnate connection', default=False)
    parser.add_argument('-c', '--combine', type=bool, help='whether to use newSoftDiceLoss 1+2+4', default=False)
    parser.add_argument('-d', '--dropout', type=bool, help='whether to add one dropout layer within each DownSampling operation', default=False)
    parser.add_argument('-f', '--flooding', type=bool, help='whether to apply flooding strategy during training', default=False)
    parser.add_argument('--seglabel', type=int, help='whether to train the model with 1 or all 3 labels', default=0)
    parser.add_argument('-t', '--act', type=int, help='activation function, choose between 0 and 1; 0-ReLU; 1-Sin', default=0)
    parser.add_argument('-s', '--save_folder', type=str, help='The folder for model saving', default='saved_pth')
    parser.add_argument('-p', '--pth', type=str, help='name of the saved pth file', default='')
    parser.add_argument('-i', '--image_shape', type=int,  nargs='+', help='The shape of input tensor;'
                                                                    'have to be dividable by 16 (H, W, D)',
                        default=[128, 192, 160])

    return parser.parse_args()

args = init_args()
num_epoch = args.epoch
num_gpu = args.num_gpu
batch_size = 2
# batch_size = 1
new_SoftDiceLoss = args.combine
dropout = args.dropout
flooding = args.flooding
seglabel_idx = args.seglabel
label_list = [None, "WT", "TC", "ET"]   # None represents using all 3 labels
dice_list = [None, "dice_wt", "dice_tc", "dice_et"]
seg_label = label_list[seglabel_idx]  # used for data generation
seg_dice = dice_list[seglabel_idx]  # used for dice calculation
activation_list = ["relu", "sin"]
activation = activation_list[args.act]
save_folder = args.save_folder
pth_name = args.pth
image_shape = tuple(args.image_shape)

concat = args.concat
attention_idx = args.attention

attention = True

config["cuda_devices"] = True
if num_gpu == 0:
    config["cuda_devices"] = None
elif num_gpu == 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
elif num_gpu == 2:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
elif num_gpu == 4:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

config["batch_size"] = batch_size
config["validation_batch_size"] = batch_size
config["model_name"] = "UNetVAE-bs{}".format(config["batch_size"])  # logger
config["image_shape"] = image_shape
config["activation"] = activation
config["input_shape"] = tuple([config["batch_size"]] + [config["nb_channels"]] + list(config["image_shape"]))
config["result_path"] = os.path.join(config["base_path"], "models")   # save models and status files
# config["saved_model_file"] = '/Users/missshihonghowru/Desktop/nyu master/brats-challenge/pth-models_status/model.pth'
config["saved_model_file"] = config["result_path"] + pth_name
config["overwrite"] =True# True
if pth_name:
    config["overwrite"] = False
config["epochs"] = int(num_epoch)
config["attention"] = attention
config["seg_label"] = seg_label                             # used for data generation
config["num_labels"] = 1 if config["seg_label"] else 3      # used for model constructing
config["seg_dice"] = seg_dice                               # used for dice calculation
config["new_SoftDiceLoss"] = new_SoftDiceLoss
config["flooding"] = flooding
if config["flooding"]:
    config["flooding_level"] = 0.15


def main():
    # init or load model
    print("init model with input shape", config["input_shape"])

    model = SFNet(config=config)

    log_var_a = torch.zeros((1,)).cuda()
    log_var_a.requires_grad = True

    parameters = ([p for p in model.parameters()] + [log_var_a])

    # parameters = model.parameters()
    optimizer = optim.Adam(parameters, 
                           lr=config["initial_learning_rate"],
                           weight_decay=config["L2_norm"])
    start_epoch = 1

    if config["enable"]:
        loss_function = CombinedLoss(combine=True,
                                     k1=config["loss_k1_weight"], k2=config["loss_k2_weight"],alpha=config["focal_alpha"], gamma=config["focal_gamma"], var_a=log_var_a)#, var_a=log_var_a, var_b=log_var_b
    else:
        loss_function = SoftDiceLoss(combine=config["combine"])


    with open('valid_list.txt', 'r') as f:
        val_list = f.read().splitlines()

    with open('train_list.txt','r') as f:
        tr_list = f.read().splitlines()

    config["training_patients"] = tr_list
    config["validation_patients"] = val_list
    # data_generator
    print("data generating")
    training_data = BratsDataset(phase="train", config=config)
    # x = training_data[0] # for test
    valildation_data = BratsDataset(phase="validate", config=config)
    train_logger = Logger(model_name=config["model_name"] + '.h5',
                          header=['epoch', 'loss', 'wt-dice', 'tc-dice', 'et-dice', 'lr'])

    if not config["overwrite"] and config["saved_model_file"] is not None:
        if not os.path.exists(config["saved_model_file"]):
            raise Exception("Invalid model path!")
        model, start_epoch, optimizer_resume = load_old_model(model, optimizer, saved_model_path='')
        parameters = model.parameters()
        optimizer = optim.Adam(parameters,
                               lr=optimizer_resume.param_groups[0]["lr"],
                               weight_decay=optimizer_resume.param_groups[0]["weight_decay"])

    if config["cuda_devices"] is not None:
        model = model.cuda()
        loss_function = loss_function.cuda()
        model = nn.DataParallel(model)    # multi-gpu training
        for state in optimizer.state.values():
	        for k, v in state.items():
	            if isinstance(v, torch.Tensor):
	                state[k] = v.cuda()

    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=config["lr_decay"], patience=config["patience"])
    # scheduler = lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=poly_lr_scheduler)  # can't restore lr correctly
    # log = pd.DataFrame(index=[], columns=['a'])
    max_val_WT_dice = 0.
    max_val_AVG_dice = 0.
    for i in range(start_epoch, config["epochs"]):
        train_epoch(epoch=i, 
                    data_set=training_data, 
                    model=model,
                    criterion=loss_function, 
                    optimizer=optimizer, 
                    opt=config, 
                    logger=train_logger) 
        
        val_loss, WT_dice, TC_dice, ET_dice= val_epoch(epoch=i,
                    data_set=valildation_data,
                    model=model,
                    criterion=loss_function,
                    opt=config,
                    optimizer=optimizer,
                    logger=train_logger)

        # tmp = pd.Series([
        #     vara.item(),
        #     # varb.item(),
        #
        # ], index=['a'])
        # log = log.append(tmp, ignore_index=True)
        # log.to_csv('/media/gdlls/data/mfh/BRATS/model/log_var.csv', index=False)
        # scheduler.step()
        # scheduler.step(val_loss)
        dices = np.array([WT_dice, TC_dice, ET_dice])
        AVG_dice = dices.mean()
        # if config["checkpoint"] and (WT_dice > max_val_WT_dice or AVG_dice > max_val_AVG_dice or ET_dice > max_val_ET_dice or TC_dice > max_val_TC_dice ):
        if config["checkpoint"] :
            # max_val_WT_dice = WT_dice
            # max_val_TC_dice = TC_dice
            # max_val_ET_dice = ET_dice
            # max_val_AVG_dice = AVG_dice

            # save_dir = os.path.join(config["result_path"], config["model_file"].split("/")[-1].split(".h5")[0])
            save_dir = config["result_path"]
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            # save_states_path = os.path.join(save_dir, 'e{0}W{1:.3f}A:{2:.3f}.pth'.format(i, WT_dice, AVG_dice))
            save_states_path = os.path.join(save_dir,'epoch_{0}_AVGDice_{1:.4f}.pth'.format(i,AVG_dice))
            if config["cuda_devices"] is not None:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            states = {
                'epoch': i,
                'state_dict': state_dict,
                'optimizer': optimizer.state_dict(),
            }
            torch.save(states, save_states_path)
            save_model_path = os.path.join(save_dir, "best_model.pth")
            if os.path.exists(save_model_path):
                os.system("rm "+save_model_path)
            torch.save(model, save_model_path)
        # print("batch {0:d} finished, validation loss:{1:.4f}; WTDice:{2:.4f},TCDice:{3:.4f},ETDice:{4:.4f},AVGDice:{5:.4f}".format(i, val_loss, WT_dice,TC_dice,ET_dice, AVG_dice))
        # print(i, val_loss, WT_dice, TC_dice, ET_dice, AVG_dice)
        print("batch {0:d} finished,WTDice:{1:.4f};TC_dice:{2:.4f};ET_dice:{3:.4f},AVGDice:{4:.4f}".format(i,  WT_dice, TC_dice, ET_dice, AVG_dice))
if __name__ == '__main__':
    main()
