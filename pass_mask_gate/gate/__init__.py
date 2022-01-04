from ..config import *
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import numpy as np
from enum import Enum, unique


def show_mask(mask, index=1):
    """
    Given mask produce a plt graph for visualization.

    if channel num is larger than 1, please choose one layer to print
    by setting index. Batch size over 1 is not acceptable please set
    the index in the first dimension
    """
    # feature_map = 1-feature_map
    mask = mask.squeeze(0)
    mask = mask.cpu().numpy()
    feature_map_num = mask.shape[0]
    row_num = np.ceil(np.sqrt(feature_map_num))
    plt.figure()
    # for index in range(1, feature_map_num+1):
    # plt.subplot(row_num, row_num, index)
    plt.imshow(mask[index - 1], cmap='gray')
    plt.axis('off')
    plt.show()
    # plt.savefig(cfg.vis_dir + '/featuremap'+'.jpg')
    plt.close()


class LearnableBernoulliRounding(torch.autograd.Function):
    """
    Generate a 0-1 mask obey Bernoulli distribution

    You can force add skip_rate to maintain a lower bound,
    accuracy drops if you set that too high.
    Negative allowed for skip_rate parameter if you want better accuracy.
    """
    @staticmethod
    def forward(ctx, input, skip_rate=0):
        y = input + skip_rate
        y = torch.clip(y, min=0.0, max=1.0)
        y = torch.bernoulli(y)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        # print("====== Go BernoulliRounding backward")
        return grad_output, None


class BernoulliRounding(torch.autograd.Function):
    """
    Generate a 0-1 mask obey Bernoulli distribution
    """
    @staticmethod
    def forward(ctx, input):
        y = input.clone()
        y = torch.clip(y, min=0.0, max=1.0)
        y = torch.bernoulli(y)

        return y

    @staticmethod
    def backward(ctx, grad_output):
        # print("====== Go BernoulliRounding backward")
        return grad_output, None


class HardSoftmax(torch.autograd.Function):
    """
    Generate a 0-1 mask simply by rounding
    """
    @staticmethod
    def forward(ctx, input):
        y_hard = input
        y_hard = y_hard.zero_()
        y_hard[input >= 0.5] = 1
        return y_hard

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class GumbelSigmoid(torch.nn.Module):
    """
    Implementation of gumbel softmax for a binary case using gumbel sigmoid.
    """
    def __init__(self):
        super(GumbelSigmoid, self).__init__()
        self.running_mode = None
        self.sigmoid = nn.Sigmoid()


    # @staticmethod
    def normalize_to_one(self, logits):
        logits_max = torch.max(logits)
        logits_min = torch.min(logits)
        gap = logits_max - logits_min
        logits = torch.div(logits - logits_min, gap)
        return logits

    # @staticmethod
    def sample_gumbel_like(self, template_tensor, eps=1e-10):
        uniform_samples_tensor = template_tensor.clone().uniform_()
        gumbel_samples_tensor = -torch.log(
            eps - torch.log(uniform_samples_tensor + eps)
        )
        return gumbel_samples_tensor

    def gumbel_sigmoid_sample(self, logits, temperature):
        """
        Adds noise to the logits and takes the sigmoid. No Gumbel noise during inference.
        """
        if self.running_mode == RunningMode.GatePreTrain or self.running_mode == RunningMode.FineTuning:
            gumbel_samples_tensor = self.sample_gumbel_like(logits.data)
            gumbel_trick_log_prob_samples = logits + gumbel_samples_tensor.data
        else:
            gumbel_trick_log_prob_samples = logits
        # soft_samples = self.sigmoid(gumbel_trick_log_prob_samples / temperature)
        soft_samples = self.normalize_to_one(gumbel_trick_log_prob_samples / temperature)

        return soft_samples

    def gumbel_sigmoid(self, logits, temperature=2 / 3, hard=False):
        out = self.gumbel_sigmoid_sample(logits, temperature)
        if hard:
            out = HardSoftmax.apply(out)
        else:
            out = BernoulliRounding.apply(out)
            # out = LearnableBernoulliRounding.apply(out, self.skip_rate)
        return out

    def forward(self, logits, running_mode, temperature=2 / 3):
        self.running_mode = running_mode
        if self.running_mode == RunningMode.GatePreTrain or self.running_mode == RunningMode.FineTuning:
            return self.gumbel_sigmoid(logits, temperature=temperature, hard=False)
        else:
            return self.gumbel_sigmoid(logits, temperature=temperature, hard=True)


class Residual(nn.Module):
    def __init__(self, fn):
        super(Residual, self).__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class MaskGate(nn.Module):
    def __init__(self, patch_size, patch_num, input_channel_num, hidden_dim, dim=256, depth=4, kernel_size=9, fc_dim=4096, fc_depth=0):
        super(MaskGate, self).__init__()
        # output should be patch_num * patch_num tensor mask
        self.hidden_dim = hidden_dim
        self.input_channel_num = input_channel_num
        self.patch_num = patch_num
        self.patch_size = patch_size
        self.convmixer = nn.Sequential(
            nn.Conv2d(self.input_channel_num, dim, kernel_size=patch_size, stride=patch_size),
            nn.GELU(),
            nn.BatchNorm2d(dim),
            *[nn.Sequential(Residual(nn.Sequential(nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                                                   nn.GELU(),
                                                   nn.BatchNorm2d(dim)
                                                   )),
                            nn.Conv2d(dim, dim, kernel_size=1),
                            nn.GELU(),
                            nn.BatchNorm2d(dim)
                            ) for i in range(depth)],
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(dim, fc_dim),
            *[nn.Linear(fc_dim, fc_dim) for i in range(fc_depth)],
            nn.Linear(fc_dim, self.patch_num)
        )
        self.sigmoid = GumbelSigmoid()

    def forward(self, now_input, previous_input, mask_mode, running_mode):
        current_input = torch.cat((now_input, previous_input), 1)
        mask = self.convmixer(current_input)
        mask = self.sigmoid(mask, running_mode)

        h, w = now_input.shape[2:]
        b = now_input.shape[0]
        mask_h = h // self.patch_size
        mask_w = w // self.patch_size

        mask = torch.reshape(mask, (mask.shape[0], 1, mask_h, mask_w))

        # mask = torch.zeros_like(mask)
        # mask = torch.ones_like(mask)
        # mask = torch.bernoulli(mask * 0.02)
        # mask = 1 - mask

        if mask_mode == MaskMode.Positive:
            current_skip = torch.count_nonzero(mask)
            current_total_patch = self.patch_num * b
            current_skip_rate = current_skip / current_total_patch
            h_out = h
            w_out = w
            current_mac = b * h_out * w_out * self.hidden_dim
            cfg.total_mac += current_mac
            cfg.skipped_mac += current_mac * current_skip_rate
            # mac_gates = n * h_out * w_out * c_in * 1 * k_h * k_w (n = batch_size, h_out = (h + 2*p - k)/s + 1 = h + 0 - 1 / 1
            cfg.skipped_patch += current_skip
            cfg.total_patch += current_total_patch


        mask = mask.expand(now_input.shape[0], now_input.shape[1], mask_h, mask_w)

        m = torch.nn.Upsample(scale_factor=self.patch_size, mode='nearest')
        mask = m(mask)
        # if cfg.mode == RunningMode.Test:
        # show_mask(mask.detach())

        if mask_mode == MaskMode.Positive:
            now_input = torch.mul(previous_input, mask) + torch.mul(now_input, 1 - mask)
        if mask_mode == MaskMode.Negative:
            now_input = torch.mul(now_input, mask) + torch.mul(previous_input, 1 - mask)

        return now_input

    def init_weights(self):
        for m in self.convmixer.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.normal_(m.weight, std=0.001)
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias'):
                    if m.bias is not None:
                        # nn.init.constant_(m.bias, 0)
                        nn.init.constant_(m.bias, 0.6)
            if isinstance(m, Residual):
                for j in m.fn.modules():
                    if isinstance(j, nn.Conv2d):
                        nn.init.normal_(j.weight, std=0.001)
                        if hasattr(j, 'bias'):
                            if j.bias is not None:
                                nn.init.constant_(j.bias, 0)
                    if isinstance(j, nn.BatchNorm2d):
                        nn.init.constant_(j.weight, 1)
                        nn.init.constant_(j.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)