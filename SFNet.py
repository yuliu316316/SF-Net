
import torch
import torch.nn as nn
from siren_pytorch import Sine
import torch.nn.functional as F
def ELUCons(elu, nchan):
    if elu:
        return nn.ELU(inplace=True)
    else:
        return nn.PReLU(nchan)

# normalization between sub-volumes is necessary
# for good performance
class ContBatchNorm3d(nn.modules.batchnorm._BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))
        super(ContBatchNorm3d, self)._check_input_dim(input)

    def forward(self, input):
        #self._check_input_dim(input)
        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            True, self.momentum, self.eps)
class Conv(nn.Module):
    def __init__(self,inChans, outChans, elu):
        super(Conv, self).__init__()
        self.conv1 = nn.Conv3d(inChans, outChans, kernel_size=3, stride=1, padding=1, bias=True)
#        self.conv1 = nn.Conv3d(6, 24, kernel_size=5, padding=2)
        self.bn1 = ContBatchNorm3d(outChans)
        self.relu1 = ELUCons(elu, outChans)
#        self.bn1 = ContBatchNorm3d(16)
#        self.relu1 = ELUCons(elu, 16)

    def forward(self, x):
        # do we want a PRELU here as well?
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        return out

class SKConv(nn.Module):
    def __init__(self, features, M, r, L=32):
        """ Constructor
        Args:
            features: input channel dimensionality.
            WH: input spatial dimensionality, used for GAP kernel size.
            M: the number of branchs.
            G: num of convolution groups.
            r: the radio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(SKConv, self).__init__()
        d = max(int(features/r), L)
        self.M = M
        self.features = features
        # self.gap = nn.AvgPool2d(int(WH/stride))
        #self.fc = nn.Linear(features, features)
        self.conv2 = nn.ModuleList([])
        for i in range(M):
            self.conv2.append(
                nn.Conv3d(d, features, kernel_size=1, stride=1, padding=0, bias=True)
            )
        self.softmax = nn.Softmax(dim=-1)
        self.conv1 = nn.Conv3d(features, d, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):             

        #feas = torch.cat([x1, x2], dim=1)
        fea_U = torch.sum(x, dim=-1)
#        print(fea_U.shape)
        _, mm, _, _, _ = fea_U.shape     
        fea_s = fea_U.mean(-1).mean(-1).mean(-1)
        fea_s = fea_s.unsqueeze_(dim=-1).unsqueeze_(dim=-1).unsqueeze_(dim=-1)
#        print(fea_s.shape)
        fea_z = self.conv1(fea_s)   
        for i, conv2 in enumerate(self.conv2):
            vector = conv2(fea_z).unsqueeze_(dim=-1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=-1)
#        print(attention_vectors.shape)
        attention_vectors = self.softmax(attention_vectors)
#        print(attention_vectors.shape)
        fea_v = (x * attention_vectors).sum(dim=-1)
#        print(fea_v.shape)
        return fea_v

class DownSampling(nn.Module):
    # 3x3x3 convolution and 1 padding as default
    def __init__(self, inChans, outChans, stride=2, kernel_size=3, padding=1, dropout_rate=None):
        super(DownSampling, self).__init__()

        self.dropout_flag = False
        self.conv1 = nn.Conv3d(in_channels=inChans,
                     out_channels=outChans,
                     kernel_size=kernel_size,
                     stride=stride,
                     padding=padding,
                     bias=False)
        if dropout_rate is not None:
            self.dropout_flag = True
            self.dropout = nn.Dropout3d(dropout_rate,inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        if self.dropout_flag:
            out = self.dropout(out)
        return out

class EncoderBlock(nn.Module):
    '''
    Encoder block; Green
    '''
    def __init__(self, inChans, outChans, stride=1, padding=1, num_groups=8, activation="relu", normalizaiton="group_normalization"):
        super(EncoderBlock, self).__init__()

        if normalizaiton == "group_normalization":
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=inChans)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=inChans)
        if activation == "relu":
            self.actv1 = nn.ReLU(inplace=True)
            self.actv2 = nn.ReLU(inplace=True)
        # elif activation == "elu":
        #     self.actv1 = nn.ELU(inplace=True)
        #     self.actv2 = nn.ELU(inplace=True)
        elif activation == "sin":
            self.actv1 = Sine(1.0)
            self.actv2 = Sine(1.0)
        self.conv1 = nn.Conv3d(in_channels=inChans, out_channels=outChans, kernel_size=3, stride=stride, padding=padding)
        self.conv2 = nn.Conv3d(in_channels=inChans, out_channels=outChans, kernel_size=3, stride=stride, padding=padding)


    def forward(self, x):
        residual = x

        out = self.norm1(x)
        out = self.actv1(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = self.actv2(out)
        out = self.conv2(out)

        out += residual

        return out

class LinearUpSampling(nn.Module):
    '''
    Trilinear interpolate to upsampling
    '''
    def __init__(self, inChans, outChans, scale_factor=2, mode="trilinear", align_corners=True):
        super(LinearUpSampling, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        self.conv1 = nn.Conv3d(in_channels=inChans, out_channels=outChans, kernel_size=1)
        # self.conv2 = nn.Conv3d(in_channels=inChans, out_channels=outChans, kernel_size=1)

    def forward(self, x, skipx=None):
        out = self.conv1(x)
        # out = self.up1(out)
        out = nn.functional.interpolate(out, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)

        if skipx is not None:
            out += skipx
            # out = torch.cat((out, skipx), 1)
            # out = self.conv2(out)  # Given groups=1, weight of size [128, 256, 1, 1, 1], expected input[1, 128, 32, 48, 40] to have 256 channels, but got 128 channels instead


        return out

class DecoderBlock(nn.Module):
    '''
    Decoder block
    '''
    def __init__(self, inChans, outChans, stride=1, padding=1, num_groups=8, activation="relu", normalizaiton="group_normalization"):
        super(DecoderBlock, self).__init__()

        if normalizaiton == "group_normalization":
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=outChans)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=outChans)
        if activation == "relu":
            self.actv1 = nn.ReLU(inplace=True)
            self.actv2 = nn.ReLU(inplace=True)
        # elif activation == "elu":
        #     self.actv1 = nn.ELU(inplace=True)
        #     self.actv2 = nn.ELU(inplace=True)
        elif activation == "sin":
            self.actv1 = Sine(1.0)
            self.actv2 = Sine(1.0)
        self.conv1 = nn.Conv3d(in_channels=inChans, out_channels=outChans, kernel_size=3, stride=stride, padding=padding)
        self.conv2 = nn.Conv3d(in_channels=outChans, out_channels=outChans, kernel_size=3, stride=stride, padding=padding)


    def forward(self, x):
        residual = x

        out = self.norm1(x)
        out = self.actv1(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = self.actv2(out)
        out = self.conv2(out)

        out += residual

        return out

class OutputTransition(nn.Module):
    '''
    Decoder output layer
    output the prediction of segmentation result
    '''
    def __init__(self, inChans, outChans):
        super(OutputTransition, self).__init__()

        self.conv1 = nn.Conv3d(in_channels=inChans, out_channels=outChans, kernel_size=1)
        self.actv1 = torch.sigmoid

    def forward(self, x):
        return self.actv1(self.conv1(x))

class OutputTransition1(nn.Module):
    '''
    Decoder output layer
    output the prediction of segmentation result
    '''
    def __init__(self, inChans, outChans):
        super(OutputTransition1, self).__init__()

        self.conv1 = nn.Conv3d(in_channels=inChans, out_channels=outChans, kernel_size=1)

    def forward(self, x):
        return self.conv1(x)

class VDResampling(nn.Module):
    '''
    Variational Auto-Encoder Resampling block
    '''
    def __init__(self, inChans=256, outChans=256, dense_features=(10, 12, 8), stride=2, kernel_size=3, padding=1,
                 activation="relu", normalizaiton="group_normalization"):
        super(VDResampling, self).__init__()

        midChans = int(inChans / 2)
        self.dense_features = dense_features
        if normalizaiton == "group_normalization":
            self.gn1 = nn.GroupNorm(num_groups=8, num_channels=inChans)
        if activation == "relu":
            self.actv1 = nn.ReLU(inplace=True)
            self.actv2 = nn.ReLU(inplace=True)
        # elif activation == "elu":
        #     self.actv1 = nn.ELU(inplace=True)
        #     self.actv2 = nn.ELU(inplace=True)
        elif activation == "sin":
            self.actv1 = Sine(1.0)
            self.actv2 = Sine(1.0)
        self.conv1 = nn.Conv3d(in_channels=inChans, out_channels=16, kernel_size=kernel_size, stride=stride, padding=padding)
        self.dense1 = nn.Linear(in_features=16*dense_features[0]*dense_features[1]*dense_features[2], out_features=256)
        self.dense2 = nn.Linear(in_features=128, out_features=128*dense_features[0]*dense_features[1]*dense_features[2])
        self.up0 = LinearUpSampling(128, outChans)

    def forward(self, x):
        out = self.gn1(x)
        out = self.actv1(out)
        out = self.conv1(out)   # 16*10*12*8  # 16, 8, 12, 12
        out = out.view(-1, self.num_flat_features(out))  # flatten  16*8*12*12
        out_vd = self.dense1(out)
        distr = out_vd
        out = VDraw(out_vd)  # 128
        out = self.dense2(out)
        out = self.actv2(out)
        out = out.view((-1, 128, self.dense_features[0], self.dense_features[1], self.dense_features[2]))  # flat to conv
        # out = out.view((1, 128, self.dense_features[0], self.dense_features[1], self.dense_features[2]))
        out = self.up0(out)  # include conv1 and upsize 256*20*24*16

        return out, distr

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s

        return num_features

def VDraw(x):
    # Generate a Gaussian distribution with the given mean(128-d) and std(128-d)
    return torch.distributions.Normal(x[:, :128], x[:, 128:]).sample()

class VDecoderBlock(nn.Module):
    '''
    Variational Decoder block
    '''
    def __init__(self, inChans, outChans, activation="relu", normalizaiton="group_normalization", mode="trilinear"):
        super(VDecoderBlock, self).__init__()

        self.up0 = LinearUpSampling(inChans, outChans, mode=mode)
        self.block = DecoderBlock(outChans, outChans, activation=activation, normalizaiton=normalizaiton)

    def forward(self, x):
        out = self.up0(x)
        out = self.block(out)

        return out

class VAE(nn.Module):
    '''
    Variational Auto-Encoder : to group the features extracted by Encoder
    '''
    def __init__(self, inChans=256, outChans=4, dense_features=(10, 12, 8),
                 activation="relu", normalizaiton="group_normalization", mode="trilinear"):
        super(VAE, self).__init__()

        self.vd_resample = VDResampling(inChans=inChans, outChans=inChans, dense_features=dense_features)
        self.vd_block2 = VDecoderBlock(inChans, inChans//2)
        self.vd_block1 = VDecoderBlock(inChans//2, inChans//4)
        self.vd_block0 = VDecoderBlock(inChans//4, inChans//8)
        self.vd_end = nn.Conv3d(inChans//8, outChans, kernel_size=1)

    def forward(self, x):
        out, distr = self.vd_resample(x)
        out = self.vd_block2(out)
        out = self.vd_block1(out)
        out = self.vd_block0(out)
        out = self.vd_end(out)

        return out, distr

class SFNet(nn.Module):
    def __init__(self, config):
        super(SFNet, self).__init__()
        elu = True
        nll = False
        self.config = config
        # some critical parameters
        # self.input_shape = (2,2,16, 192, 192)
        self.seg_outChans = 3
        # self.seg_outChans = config["n_labels"]
        self.activation = "relu"
        self.normalizaiton = "group_normalization"
        self.mode = "trilinear"

        # Encoder Blocks
        # self.conv1 = Conv(1,8, elu)
        # self.conv2 = Conv(8,8, elu)
        # self.sknet = SKConv(8,2,2)


        self.in_conv0 = DownSampling(inChans=3, outChans=32, stride=1, dropout_rate=0.2)
        self.en_block0 = EncoderBlock(32, 32, activation=self.activation, normalizaiton=self.normalizaiton)
        self.en_down1 = DownSampling(32, 64)
        self.en_block1_0 = EncoderBlock(64, 64, activation=self.activation, normalizaiton=self.normalizaiton)
        self.en_block1_1 = EncoderBlock(64, 64, activation=self.activation, normalizaiton=self.normalizaiton)
        self.en_down2 = DownSampling(64, 128)
        self.en_block2_0 = EncoderBlock(128, 128, activation=self.activation, normalizaiton=self.normalizaiton)
        self.en_block2_1 = EncoderBlock(128, 128, activation=self.activation, normalizaiton=self.normalizaiton)
        self.en_down3 = DownSampling(128, 256)
        self.en_block3_0 = EncoderBlock(256, 256, activation=self.activation, normalizaiton=self.normalizaiton)
        self.en_block3_1 = EncoderBlock(256, 256, activation=self.activation, normalizaiton=self.normalizaiton)
        self.en_block3_2 = EncoderBlock(256, 256, activation=self.activation, normalizaiton=self.normalizaiton)
        self.en_block3_3 = EncoderBlock(256, 256, activation=self.activation, normalizaiton=self.normalizaiton)

        # Decoder Blocks
        self.de_up2 = LinearUpSampling(256, 128, mode=self.mode)
        self.de_block2 = DecoderBlock(128, 128, activation=self.activation, normalizaiton=self.normalizaiton)
        self.de_up1 = LinearUpSampling(128, 64, mode=self.mode)
        self.de_block1 = DecoderBlock(64, 64, activation=self.activation, normalizaiton=self.normalizaiton)
        self.de_up0 = LinearUpSampling(64, 32, mode=self.mode)
        self.de_block0 = DecoderBlock(32, 32, activation=self.activation, normalizaiton=self.normalizaiton)
        self.de_end = OutputTransition(32, self.seg_outChans)

        # Variational Auto-Encoder
        self.de_up22 = LinearUpSampling(256, 128, mode=self.mode)
        self.de_block22 = DecoderBlock(128, 128, activation=self.activation, normalizaiton=self.normalizaiton)
        self.de_up12 = LinearUpSampling(128, 64, mode=self.mode)
        self.de_block12 = DecoderBlock(64, 64, activation=self.activation, normalizaiton=self.normalizaiton)
        self.de_up02 = LinearUpSampling(64, 32, mode=self.mode)
        self.de_block02 = DecoderBlock(32, 32, activation=self.activation, normalizaiton=self.normalizaiton)
        self.de_end_1 = OutputTransition(32, 2)
        # self.dense_features = (self.input_shape[2]//16, self.input_shape[3]//16, self.input_shape[4]//16)  # 8, 12, 12
        # self.vae = VAE(256, outChans=self.inChans, dense_features=self.dense_features)

    def forward(self, x):

        out_init = self.in_conv0(x)  # 32, 128, 192, 192
        out_en0 = self.en_block0(out_init)  # 32, 128, 192, 192
        out_en1 = self.en_block1_1(self.en_block1_0(self.en_down1(out_en0)))  # 64, 64, 96, 96
        out_en2 = self.en_block2_1(self.en_block2_0(self.en_down2(out_en1)))  # 128, 32, 48, 48
        out_en3 = self.en_block3_3(
            self.en_block3_2(
                self.en_block3_1(
                    self.en_block3_0(
                        self.en_down3(out_en2)))))  # 256, 16, 24, 24

        out_de2 = self.de_block2(self.de_up2(out_en3, out_en2))   # forward
        out_de1 = self.de_block1(self.de_up1(out_de2, out_en1))
        out_de0 = self.de_block0(self.de_up0(out_de1, out_en0))
        out_end = self.de_end(out_de0)

        out_fu2 = self.de_block22(self.de_up22(out_en3))   # forward
        out_fu1 = self.de_block12(self.de_up12(out_fu2))
        out_fu0 = self.de_block02(self.de_up02(out_fu1))
        out_fud = self.de_end_1(out_fu0)
#        if self.config["VAE_enable"]:
#        out_vae, out_distr = self.vae(out_en3)
        out_final = torch.cat((out_end, out_fud), 1)
#        return out_final, out_distr
#         return out_end
        return out_final
# net = NvNet(1)
# # net.apply(_init_)
#
# net = net.cuda(0)
# x = torch.randn((1, 3, 128, 192, 80)).cuda()
# # y = torch.randn((1, 1, 64, 64, 64)).cuda()
# res = net(x)
# for item in res:
#    print(item.size())