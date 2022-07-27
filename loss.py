
import torch
from torch.nn.modules.loss import _Loss 
import torch.nn as nn
import numpy as np
class FocalLoss(_Loss):
    '''
    Focal_Loss = - [alpha * (1 - p)^gamma *log(p)]  if y = 1;
               = - [(1-alpha) * p^gamma *log(1-p)]  if y = 0;
        average over batchsize; alpha helps offset class imbalance; gamma helps focus on hard samples
    '''
    def __init__(self, alpha=0.9, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, y_pred, y_true, eps=1e-8):

        alpha = self.alpha
        gamma = self.gamma
        focal_ce = - (alpha * torch.pow((1-y_pred), gamma) * torch.log(torch.clamp(y_pred, eps, 1.0)) * y_true
                      + (1-alpha) * torch.pow(y_pred, gamma) * torch.log(torch.clamp(1-y_pred, eps, 1.0)) * (1-y_true))
        focal_loss = torch.mean(focal_ce)

        return focal_loss

class SoftDiceLoss(_Loss):
    '''
    Soft_Dice = 2*|dot(A, B)| / (|dot(A, A)| + |dot(B, B)| + eps)
    eps is a small constant to avoid zero division,
    '''
    def __init__(self, combine):
        super(SoftDiceLoss, self).__init__()
        self.combine = combine

    def forward(self, y_pred, y_true, eps=1e-8):   # put 2,1,4 together

        if self.combine:
            y_pred[:, 0, :, :, :] = torch.sum(y_pred, dim=1)
            y_pred[:, 1, :, :, :] = torch.sum(y_pred[:, 1:, :, :, :], dim=1)
            y_true[:, 0, :, :, :] = torch.sum(y_true, dim=1)
            y_true[:, 1, :, :, :] = torch.sum(y_true[:, 1:, :, :, :], dim=1)

        intersection = torch.sum(torch.mul(y_pred, y_true), dim=[-3, -2, -1])
        union = torch.sum(torch.mul(y_pred, y_pred),
                          dim=[-3, -2, -1]) + torch.sum(torch.mul(y_true, y_true), dim=[-3, -2, -1]) + eps

        dice = 2 * intersection / union   # (bs, 3)
        dice_loss = 1 - torch.mean(dice)  # loss small, better
        # means = torch.mean(dice, dim=0)
        # dice_loss = 1 - 0.1*means[0] - 0.45*means[1] - 0.45*means[2]

        return dice_loss


class CustomKLLoss(_Loss):
    '''
    KL_Loss = (|dot(mean , mean)| + |dot(std, std)| - |log(dot(std, std))| - 1) / N
    N is the total number of image voxels
    '''
    def __init__(self, *args, **kwargs):
        super(CustomKLLoss, self).__init__()

    def forward(self, mean, std):
        return torch.mean(torch.mul(mean, mean)) + torch.mean(torch.mul(std, std)) - torch.mean(torch.log(torch.mul(std, std))) - 1

def tf_fspecial_gauss_3d_torch(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x_data, y_data, z_data = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1,
                             -size // 2 + 1:size // 2 + 1]

    x_data = np.expand_dims(x_data, axis=0)
    x_data = np.expand_dims(x_data, axis=1)

    y_data = np.expand_dims(y_data, axis=0)
    y_data = np.expand_dims(y_data, axis=1)

    z_data = np.expand_dims(z_data, axis=0)
    z_data = np.expand_dims(z_data, axis=1)

    x = torch.tensor(x_data, dtype=torch.float32)  
    y = torch.tensor(y_data, dtype=torch.float32)
    z = torch.tensor(z_data, dtype=torch.float32)

    g = torch.exp(-((x ** 2 + y ** 2 + z ** 2) / (3.0 * sigma ** 2)))
    return g / torch.sum(g)
class SSIM_Loss(_Loss):
    def __init__(self, *args, **kwargs):
        super(SSIM_Loss, self).__init__()

    def forward(self,img11, img2,img3, k1=0.01, k2=0.03, L=2, window_size=11): #(fusion, t1c,t2/flair)
        """
        The function is to calculate the ssim score
        """
        ones = torch.ones([2, 1, 128, 192, 160]).cuda()
        img1 = torch.mul(img11, img3) + torch.mul((ones-img11), img2)  #  fusion*t1c + (1-fusion)*t2/flair
                                                                       #ssim(fusion,tic)

        window = tf_fspecial_gauss_3d_torch(window_size, 1.5)
        window = window.cuda()
        mu1 = torch.nn.functional.conv3d(img1, window, stride=1, padding=0)
        mu2 = torch.nn.functional.conv3d(img2, window, stride=1, padding=0)
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = torch.nn.functional.conv3d(img1 * img1, window, stride=1, padding=0) - mu1_sq
        sigma2_sq = torch.nn.functional.conv3d(img2 * img2, window, stride=1, padding=0) - mu2_sq
        sigma1_2 = torch.nn.functional.conv3d(img1 * img2, window, stride=1, padding=0) - mu1_mu2
        c1 = (k1 * L) ** 2
        c2 = (k2 * L) ** 2
        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma1_2 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))

        window1 = tf_fspecial_gauss_3d_torch(window_size, 1.5)
        window1 = window1.cuda()
        mu11 = torch.nn.functional.conv3d(img1, window1, stride=1, padding=0)
        mu21 = torch.nn.functional.conv3d(img3, window1, stride=1, padding=0)
        mu1_sq1 = mu11 * mu11
        mu2_sq1 = mu21 * mu21
        mu1_mu21 = mu11 * mu21
        sigma1_sq1 = torch.nn.functional.conv3d(img1 * img1, window1, stride=1, padding=0) - mu1_sq1
        sigma2_sq1 = torch.nn.functional.conv3d(img3* img3, window1, stride=1, padding=0) - mu2_sq1
        sigma1_21 = torch.nn.functional.conv3d(img1 * img3, window1, stride=1, padding=0) - mu1_mu21
        c11 = (k1 * L) ** 2
        c21 = (k2 * L) ** 2
        ssim_map1 = ((2 * mu1_mu21 + c11) * (2 * sigma1_21 + c21)) / ((mu1_sq1 + mu2_sq1 + c11) * (sigma1_sq1 + sigma2_sq1 + c21))
        a=2*(1 - torch.mean(ssim_map))+(1-torch.mean(ssim_map1))
        # return 1-torch.mean(ssim_map)
        return a
class MSE_Loss(_Loss):
    def __init__(self, *args, **kwargs):
        super(MSE_Loss, self).__init__()

    def forward(self,img11, img2,img3):  #(fusion1, t2/flair, t1c)
        
        ones = torch.ones([2, 1, 128, 192, 160]).cuda()
        img1 = torch.mul(img11, img2) + torch.mul((ones-img11), img3)  #  fusion*t1c + (1-fusion)*t2/flair  (fusion1, t2,t1c)
        L2=nn.MSELoss()                                                               #ssim(fusion,tic)
        loss_l2_1= L2(img1,img2) #L2 T2/Flair
        loss_l2_2 = L2(img1, img3)
        loss_l2=2*loss_l2_1+loss_l2_2
        return loss_l2

class CombinedLoss(_Loss):
    
    def __init__(self, combine=True, k1=0.1, k2=0.1,alpha=0.9, gamma=2,var_a=0):#,var_a=0, var_b=0
        super(CombinedLoss, self).__init__()
        self.k1 = k1
        self.k2 = k2
        self.dice_loss = SoftDiceLoss(combine)
        # self.l2_loss = nn.MSELoss()
        self.l2_loss = MSE_Loss()
        self.kl_loss = CustomKLLoss()
        self.focal_loss = FocalLoss(alpha, gamma)
        self.ssim_loss = SSIM_Loss()
        self.var_a=var_a
        # self.var_b=var_b
        # self.var_c=var_c
    def forward(self, y_pred, y_true):
        # est_mean, est_std = (y_mid[:, :128], y_mid[:, 128:])
        seg_pred, seg_truth = (y_pred[:, :3, :, :, :], y_true[:, :3, :, :, :])  # problem
        t1c,t2,fliar=(y_true[:, 3:4, :, :, :],y_true[:, 4:5, :, :, :],y_true[:, 5:6, :, :, :])
        #
        fusion1,fusion2=(y_pred[:, 3:4, :, :, :], y_pred[:, 4:5, :, :, :])

        dice_loss = self.dice_loss(seg_pred, seg_truth)

        l2_loss_1 = self.l2_loss(fusion1, t2,t1c)
        l2_loss_2 = self.l2_loss(fusion2, fliar,t1c)
        ssim_1=self.ssim_loss(fusion1, t1c,t2)
        ssim_2 = self.ssim_loss(fusion2, t1c,fliar)




        fusion_l2= (l2_loss_1 + l2_loss_2)
        fusion_ssim = ssim_2 + ssim_1
        fusion_loss = (fusion_l2 + fusion_ssim)
        fusion_loss1 = torch.exp(-self.var_a) * fusion_loss + self.var_a


        # print("dice_loss:%.4f, fusion_l2:%.4f,fusion_ssim:%.4f，vara:%.4f, varb:%.4f" % (seg_loss1, fusion_l2,fusion_ssim,torch.exp(-self.var_a), torch.exp(-self.var_b)))
        # print("dice_loss:%.4f, fusion_loss:%.4f,vara:%.4f" % (dice_loss, fusion_loss1, torch.exp(-self.var_a)))
        return dice_loss ,fusion_loss1  #,torch.exp(-self.var_a) #,torch.exp(-self.var_b)
