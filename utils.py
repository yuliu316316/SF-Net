
import pickle

import tensorboardX
import os
from collections import OrderedDict
import SimpleITK as sitk
import torch
from tqdm import tqdm
import sys
sys.path.append(".")
from utils import AverageMeter, calculate_accuracy, calculate_accuracy_singleLabel
from torch.utils.data import Dataset

import numpy as np

np.random.seed(0)

import random

random.seed(0)
def validation_sampling(data_list, test_size=0.2):
    n = len(data_list)
    m = int(n * test_size)
    val_items = random.sample(data_list, m)
    tr_items = list(set(data_list) - set(val_items))
    return tr_items, val_items


def random_intensity_shift(imgs_array, brain_mask, limit=0.1):
    """
    Only do intensity shift on brain voxels
    :param imgs_array: The whole input image with shape of (4, 155, 240, 240)
    :param brain_mask:
    :param limit:
    :return:
    """

    shift_range = 2 * limit
    for i in range(len(imgs_array) - 1):
        factor = -limit + shift_range * np.random.random()
        std = imgs_array[i][brain_mask].std()
        imgs_array[i][brain_mask] = imgs_array[i][brain_mask] + factor * std
    return imgs_array


def random_scale(imgs_array, brain_mask, scale_limits=(0.9, 1.1)):
    """
    Only do random_scale on brain voxels
    :param imgs_array: The whole input image with shape of (4, 155, 240, 240)
    :param scale_limits:
    :return:
    """
    scale_range = scale_limits[1] - scale_limits[0]
    for i in range(len(imgs_array) - 1):
        factor = scale_limits[0] + scale_range * np.random.random()
        imgs_array[i][brain_mask] = imgs_array[i][brain_mask] * factor
    return imgs_array


def random_mirror_flip(imgs_array, prob=0.5):
    """
    Perform flip along each axis with the given probability; Do it for all voxels；
    labels should also be flipped along the same axis.
    :param imgs_array:
    :param prob:
    :return:
    """
    for axis in range(1, len(imgs_array.shape)):
        random_num = np.random.random()
        if random_num >= prob:
            if axis == 1:
                imgs_array = imgs_array[:, ::-1, :, :]
            if axis == 2:
                imgs_array = imgs_array[:, :, ::-1, :]
            if axis == 3:
                imgs_array = imgs_array[:, :, :, ::-1]
    return imgs_array


def random_crop(imgs_array, crop_size=(128, 192, 160), lower_limit=(0, 32, 40)):
    """
    crop the image ((155, 240, 240) for brats data) into the crop_size
    the random area is now limited at (0:155, 32:224, 40:200), by default
    :param imgs_array:
    :param crop_size:
    :return:
    """
    orig_shape = np.array(imgs_array.shape[1:])
    crop_shape = np.array(crop_size)
    # ranges = np.array(orig_shape - crop_shape, dtype=np.uint8)
    # lower_limits = np.random.randint(np.array(ranges))
    lower_limit_z = np.random.randint(lower_limit[0], 155 - crop_size[0])
    if crop_size[1] < 192:
        lower_limit_y = np.random.randint(lower_limit[1], 224 - crop_size[1])
    else:
        lower_limit_y = np.random.randint(0, 240 - crop_size[1])
    if crop_size[2] < 160:
        lower_limit_x = np.random.randint(lower_limit[2], 200 - crop_size[2])
    else:
        lower_limit_x = np.random.randint(0, 240 - crop_size[2])
    lower_limits = np.array((lower_limit_z, lower_limit_y, lower_limit_x))
    upper_limits = lower_limits + crop_shape
    imgs_array = imgs_array[:, lower_limits[0]: upper_limits[0],
                 lower_limits[1]: upper_limits[1], lower_limits[2]: upper_limits[2]]
    return imgs_array


def validation_time_crop(imgs_array, crop_size=(128, 192, 160)):
    """
    crop the image ((155, 240, 240) for brats data) into the crop_size
    :param imgs_array:
    :param crop_size:
    :return:
    """
    orig_shape = np.array(imgs_array.shape[1:])
    crop_shape = np.array(crop_size)
    lower_limit_z = np.random.randint(orig_shape[0] - crop_size[0])
    center_y = 128
    center_x = 120
    lower_limit_y = center_y - crop_size[-2] // 2  # (128, 160, 128)  (?, 48, 56)
    lower_limit_x = center_x - crop_size[-1] // 2  # (128, 192, 160)  (?, 32, 40)
    lower_limits = np.array((lower_limit_z, lower_limit_y, lower_limit_x))

    upper_limits = lower_limits + crop_shape

    imgs_array = imgs_array[:, lower_limits[0]: upper_limits[0],
                 lower_limits[1]: upper_limits[1], lower_limits[2]: upper_limits[2]]
    return imgs_array


def test_time_crop(imgs_array, crop_size=(144, 192, 192)):
    """
    crop the test image around the center; default crop_zise change from (128, 192, 160) to (144, 192, 160)
    :param imgs_array:
    :param crop_size:
    :return: image with the size of crop_size
    """
    orig_shape = np.array(imgs_array.shape[1:])
    crop_shape = np.array(crop_size)
    center = orig_shape // 2
    lower_limits = center - crop_shape // 2  # (13, 24, 40) (5, 24, 40)
    upper_limits = center + crop_shape // 2  # (141, 216, 200) (149, 216, 200）
    # upper_limits = lower_limits + crop_shape
    imgs_array = imgs_array[:, lower_limits[0]: upper_limits[0],
                 lower_limits[1]: upper_limits[1], lower_limits[2]: upper_limits[2]]
    return imgs_array


def preprocess_label(img, single_label=None):
    """
    Separates out the 3 labels from the segmentation provided, namely:
    GD-enhancing tumor (ET — label 4), the peritumoral edema (ED — label 2))
    and the necrotic and non-enhancing tumor core (NCR/NET — label 1)
    """

    ncr = img == 1  # Necrotic and Non-Enhancing Tumor (NCR/NET) - orange
    ed = img == 2  # Peritumoral Edema (ED) - yellow
    et = img == 4  # GD-enhancing Tumor (ET) - blue
    # print("ed",et.shape)
    if not single_label:
        # return np.array([ncr, ed, et], dtype=np.uint8)
        return np.array([ed, ncr, et], dtype=np.uint8)
    elif single_label == "WT":
        img[ed] = 1
        img[et] = 1
    elif single_label == "TC":
        img[ncr] = 0
        img[ed] = 1
        img[et] = 1
    elif single_label == "ET":
        img[ncr] = 0
        img[ed] = 0
        img[et] = 1
    else:
        raise RuntimeError("the 'single_label' type must be one of WT, TC, ET, and None")
    # print("image", img.shape)
    return img[np.newaxis, :]


class BratsDataset(Dataset):
    def __init__(self, phase, config):
        super(BratsDataset, self).__init__()

        self.config = config
        self.phase = phase
        self.input_shape = config["input_shape"]
        self.data_path = config["data_path"]
        self.seg_label = config["seg_label"]
        self.intensity_shift = config["intensity_shift"]
        self.scale = config["scale"]
        self.flip = config["flip"]

        if phase == "train":
            self.patient_names = config["training_patients"]  # [:4]
        elif phase == "validate" or phase == "evaluation":
            self.patient_names = config["validation_patients"]  # [:2]
        elif phase == "test":
            self.test_path = config["test_path"]
            self.patient_names = config["test_patients"]
            self.tta_idx = config["tta_idx"]

    def __getitem__(self, index):
        patient = self.patient_names[index]
        self.file_path = os.path.join(self.data_path, 'npy', patient + ".npy")
        if self.phase == "test":
            self.file_path = os.path.join(self.test_path, 'npy', patient + ".npy")
        imgs_npy = np.load(self.file_path)

        if self.phase == "train":
            # nonzero_masks = [i != 0 for i in imgs_npy[:-1]]
            nonzero_masks = [i != 0 for i in imgs_npy[0:3]]
            brain_mask = np.zeros(imgs_npy.shape[1:], dtype=bool)

            for chl in range(len(nonzero_masks)):
                brain_mask = brain_mask | nonzero_masks[chl]  # (155, 240, 240)
            # data augmentation
            cur_image_with_label = imgs_npy.copy()
            # cur_image = cur_image_with_label[:-1]
            cur_image = cur_image_with_label[0:3]
            if self.intensity_shift:
                cur_image = random_intensity_shift(cur_image, brain_mask)
            if self.scale:
                cur_image = random_scale(cur_image, brain_mask)

            # cur_image_with_label[:-1] = cur_image
            cur_image_with_label[0:3] = cur_image
            cur_image_with_label = random_crop(cur_image_with_label, crop_size=self.input_shape[2:])

            if self.flip:  # flip should be performed with labels
                cur_image_with_label = random_mirror_flip(cur_image_with_label)

        elif self.phase == "validate":
            # cur_image_with_label = validation_time_crop(imgs_npy)
            cur_image_with_label = validation_time_crop(imgs_npy, crop_size=self.input_shape[2:])

        elif self.phase == "evaluation":
            cur_image_with_label = imgs_npy.copy()

        if self.phase == "validate" or self.phase == "train" or self.phase == "evaluation":
            inp_data = cur_image_with_label[0:3]
            source_data= cur_image_with_label[4:7]
            # seg_label = preprocess_label(cur_image_with_label[-1], "WT")
            seg_label = preprocess_label(cur_image_with_label[3], self.seg_label)
            # print("seg_label",seg_label.shape)
            final_label = np.concatenate((seg_label, source_data), axis=0)
            # print("seg_label", final_label.shape)
            # final_label = seg_label

            return np.array(inp_data), np.array(final_label)



    # np.array() solve the problem of "ValueError: some of the strides of a given numpy array are negative"

    def __len__(self):
        return len(self.patient_names)
def val_epoch(epoch, data_set, model, criterion, optimizer, opt, logger):
    print('validation at epoch {}'.format(epoch))

    model.eval()

    losses = AverageMeter()
    WT_dice = AverageMeter()
    TC_dice = AverageMeter()
    ET_dice = AverageMeter()

    valildation_loader = torch.utils.data.DataLoader(dataset=data_set,
                                                     batch_size=opt["validation_batch_size"],
                                                     shuffle=False,
                                                     pin_memory=True)
    val_process = tqdm(valildation_loader)
    for i, (inputs, targets) in enumerate(val_process):
        if i > 0:
            val_process.set_description("Epoch:%d;Loss:%.4f; dice-WT:%.4f, lr: %.6f" % (epoch,
                                                                                        losses.avg.item(),
                                                                                        WT_dice.avg.item(),
                                                                                        optimizer.param_groups[0][
                                                                                            'lr']))
            # val_process.set_description("Epoch:%d;Loss:%.4f; dice-WT:%.4f, TC:%.4f, ET:%.4f, lr: %.6f"%(epoch,
            #                                  losses.avg.item(), WT_dice.avg.item(), TC_dice.avg.item(),
            #                                  ET_dice.avg.item(), optimizer.param_groups[0]['lr']))
        if opt["cuda_devices"] is not None:
            # targets = targets.cuda(async=True)
            inputs = inputs.type(torch.FloatTensor)
            inputs = inputs.cuda()
            targets = targets.type(torch.FloatTensor)
            targets = targets.cuda()
        with torch.no_grad():
            # if opt["VAE_enable"]:
            #     outputs, distr = model(inputs)
            #     loss = criterion(outputs, targets, distr)
            # else:
            #     outputs = model(inputs)
            #     loss = criterion(outputs, targets)
            outputs = model(inputs)
            seg_loss, fusion_loss = criterion(outputs, targets)
            loss = seg_loss + fusion_loss
            # loss = seg_loss
        # acc, sum_ = calculate_accuracy(outputs.cpu(), targets.cpu())
        acc = calculate_accuracy(outputs.cpu(), targets.cpu())

        losses.update(loss.cpu(), inputs.size(0))
        WT_dice.update(acc["dice_wt"], inputs.size(0))
        TC_dice.update(acc["dice_tc"], inputs.size(0))
        ET_dice.update(acc["dice_et"], inputs.size(0))

    logger.log(phase="val", values={
        'epoch': epoch,
        'loss': format(losses.avg.item(), '.4f'),
        'wt-dice': format(WT_dice.avg.item(), '.4f'),
        'tc-dice': format(TC_dice.avg.item(), '.4f'),
        'et-dice': format(ET_dice.avg.item(), '.4f'),
        'lr': optimizer.param_groups[0]['lr']
    })
    return losses.avg, WT_dice.avg, TC_dice.avg, ET_dice.avg
    # return losses.avg, WT_dice.avg

def train_epoch(epoch, data_set, model, criterion, optimizer, opt, logger, extra_train=True):  # False
    print('train at epoch {}'.format(epoch))

    model.train()

    losses = AverageMeter()
    WT_dice = AverageMeter()
    TC_dice = AverageMeter()
    ET_dice = AverageMeter()

    # data_set.file_open()
    train_loader = torch.utils.data.DataLoader(dataset=data_set,
                                               batch_size=opt["batch_size"],
                                               shuffle=True,
                                               pin_memory=True)
    training_process = tqdm(train_loader)
    for i, (inputs, targets) in enumerate(training_process):
        if i > 0:
            training_process.set_description("Epoch:%d;Loss:%.4f; dice-WT:%.4f, TC:%.4f, ET:%.4f, lr: %.6f" % (epoch,
                                                                                                               losses.avg.item(),
                                                                                                               WT_dice.avg.item(),
                                                                                                               TC_dice.avg.item(),
                                                                                                               ET_dice.avg.item(),
                                                                                                               optimizer.param_groups[
                                                                                                                   0][
                                                                                                                   'lr']))
            # training_process.set_description("Epoch:%d;Loss:%.4f; dice-WT:%.4f, lr: %.6f" % (epoch,
            #                                 losses.avg.item(),WT_dice.avg.item(),
            #                                 optimizer.param_groups[0]['lr']))
        if opt["cuda_devices"] is not None:
            inputs = inputs.type(torch.FloatTensor)
            inputs = inputs.cuda()
            targets = targets.type(torch.FloatTensor)
            targets = targets.cuda()
            # print("inputs",inputs.shape)
            # print("tagerts",targets.shape)
            # t2 = inputs[:, 0, :, :, :]
            # t2_ = t2.unsqueeze_(dim=1)
            # flair = inputs[:, 0, :, :, :]
            # flair_ = flair.unsqueeze_(dim=1)
        # if opt["VAE_enable"]:
        #     outputs, distr = model(inputs)
        #     loss = criterion(outputs, targets, distr)
        # else:

        outputs = model(inputs)

        seg_loss, fusion_loss = criterion(outputs, targets)
        loss = seg_loss + fusion_loss
        # loss = seg_loss

        if opt["flooding"]:
            b = opt["flooding_level"]
            loss = (loss - b).abs() + b  # flooding

        if not opt["seg_dice"]:
            acc = calculate_accuracy(outputs.cpu(), targets.cpu())  # dice_coefficient
        else:
            acc = dict()
            acc["dice_wt"] = torch.tensor(0)
            acc["dice_tc"] = torch.tensor(0)
            acc["dice_et"] = torch.tensor(0)
            singleLabel_acc = calculate_accuracy_singleLabel(outputs.cpu(), targets.cpu())
            acc[opt["seg_dice"]] = singleLabel_acc

        losses.update(loss.cpu(), inputs.size(0))  # batch_avg
        WT_dice.update(acc["dice_wt"], inputs.size(0))
        TC_dice.update(acc["dice_tc"], inputs.size(0))
        ET_dice.update(acc["dice_et"], inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    logger.log(phase="train", values={
        'epoch': epoch,
        'loss': format(losses.avg.item(), '.4f'),
        'wt-dice': format(WT_dice.avg.item(), '.4f'),
        'tc-dice': format(TC_dice.avg.item(), '.4f'),
        'et-dice': format(ET_dice.avg.item(), '.4f'),
        'lr': optimizer.param_groups[0]['lr']
    })

    if extra_train:
        return losses.avg.item(), WT_dice.avg.item(), TC_dice.avg.item(), ET_dice.avg.item()
        # return losses.avg.item(), WT_dice.avg.item()


def pickle_load(in_file):
    with open(in_file, "rb") as opened_file:
        return pickle.load(opened_file)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):

    def __init__(self, model_name, header):
        self.header = header
        self.writer = tensorboardX.SummaryWriter("./runs/"+model_name.split("/")[-1].split(".h5")[0])

    def __del(self):
        self.writer.close()

    def log(self, phase, values):
        epoch = values['epoch']
        
        for col in self.header[1:]:
            self.writer.add_scalar(phase+"/"+col, float(values[col]), int(epoch))


def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))

    return value


def combine_labels(labels):
    """
    Combine wt, tc, et into WT; tc, et into TC; et into ET
    :param labels: torch.Tensor of size (bs, 3, ?,?,?); ? is the crop size
    :return:
    """
    whole_tumor = labels[:, :3, :, :, :].sum(1)  # could have 2 or 3
    tumor_core = labels[:, 1:3, :, :, :].sum(1)
    enhanced_tumor = labels[:, 2:3, :, :, :].sum(1)
    whole_tumor[whole_tumor != 0] = 1
    tumor_core[tumor_core != 0] = 1
    enhanced_tumor[enhanced_tumor != 0] = 1
    return whole_tumor, tumor_core, enhanced_tumor  # (bs, ?, ?, ?)


def calculate_accuracy(outputs, targets):
    return dice_coefficient(outputs, targets)


def dice_coefficient(outputs, targets, threshold=0.5, eps=1e-8): 
    # batch_size = targets.size(0)
    y_pred = outputs[:, :3, :, :, :]  # targets[0,:3,:,:,:]
    y_truth = targets[:, :3, :, :, :]
    y_pred = y_pred > threshold
    y_pred = y_pred.type(torch.FloatTensor)
    wt_pred, tc_pred, et_pred = combine_labels(y_pred)
    wt_truth, tc_truth, et_truth = combine_labels(y_truth)
    res = dict()
    res["dice_wt"] = dice_coefficient_single_label(wt_pred, wt_truth, eps)
    res["dice_tc"] = dice_coefficient_single_label(tc_pred, tc_truth, eps)
    res["dice_et"] = dice_coefficient_single_label(et_pred, et_truth, eps)

    return res


def calculate_accuracy_singleLabel(outputs, targets, threshold=0.5, eps=1e-8):

    y_pred = outputs[:, 0, :, :, :]  # targets[0,:3,:,:,:]
    y_truth = targets[:, 0, :, :, :]
    y_pred = y_pred > threshold
    y_pred = y_pred.type(torch.FloatTensor)
    res = dice_coefficient_single_label(y_pred, y_truth, eps)
    return res


def dice_coefficient_single_label(y_pred, y_truth, eps):
    # batch_size = y_pred.size(0)
    intersection = torch.sum(torch.mul(y_pred, y_truth), dim=(-3, -2, -1)) + eps / 2  # axis=?, (bs, 1)
    union = torch.sum(y_pred, dim=(-3,-2,-1)) + torch.sum(y_truth, dim=(-3,-2,-1)) + eps  # (bs, 1)
    dice = 2 * intersection / union
    return dice.mean()
    # return dice / batch_size


def load_old_model(model, optimizer, saved_model_path, data_paralell=True):
    print("Constructing model from saved file... ")
    checkpoint = torch.load(saved_model_path, map_location='cpu')
    epoch = checkpoint["epoch"]
    # epoch = 1
    if data_paralell:
        state_dict = OrderedDict()
        for k, v in checkpoint["state_dict"].items():  # remove "module."
            if "module." in k:
                node_name = k[7:]

            else:
                node_name = k
            state_dict[node_name] = v
        model.load_state_dict(state_dict)
    else:
        model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    return model, epoch, optimizer


def combine_labels_predicting(output_array):
    """
    # (1, 3, 240, 240, 155)
    :param output_array: output of the model containing 3 seperated labels (3 channels)
    :return: res_array: conbined labels (1 channel)
    """
    shape = output_array.shape[-3:]

    if len(output_array.shape) == 5:
        bs = output_array.shape[0]
        res_array = np.zeros((bs, ) + shape)
        res_array[output_array[:, 0, :, :, :] == 1] = 2  # 1
        res_array[output_array[:, 1, :, :, :] == 1] = 1  # 2
        res_array[output_array[:, 2, :, :, :] == 1] = 4
    elif len(output_array.shape) == 4:
        res_array = np.zeros(shape)
        res_array[output_array[0, :, :, :] == 1] = 2
        res_array[output_array[1, :, :, :] == 1] = 1
        res_array[output_array[2, :, :, :] == 1] = 4
    return res_array


def dim_recovery(img_array, orig_shape=(155, 240, 240)):
    """
    used when doing inference
    :param img_array:
    :param orig_shape:
    :return:
    """
    crop_shape = np.array(img_array.shape[-3:])
    center = np.array(orig_shape) // 2
    lower_limits = center - crop_shape // 2
    upper_limits = center + crop_shape // 2
    if len(img_array.shape) == 5:
        bs, num_labels = img_array.shape[:2]
        res_array = np.zeros((bs, num_labels) + orig_shape)
        res_array[:, :, lower_limits[0]: upper_limits[0],
                        lower_limits[1]: upper_limits[1], lower_limits[2]: upper_limits[2]] = img_array
    if len(img_array.shape) == 4:
        num_labels = img_array.shape[0]
        res_array = np.zeros((num_labels, ) + orig_shape)
        res_array[:, lower_limits[0]: upper_limits[0],
                     lower_limits[1]: upper_limits[1], lower_limits[2]: upper_limits[2]] = img_array

    if len(img_array.shape) == 3:
        res_array = np.zeros(orig_shape)
        res_array[lower_limits[0]: upper_limits[0],
            lower_limits[1]: upper_limits[1], lower_limits[2]: upper_limits[2]] = img_array

    return res_array


def convert_stik_to_nparray(gz_path):
    sitkImage = sitk.ReadImage(gz_path)
    nparray = sitk.GetArrayFromImage(sitkImage)
    return nparray


def poly_lr_scheduler(epoch, num_epochs=300, power=0.9):
    return (1 - epoch/num_epochs)**power

