#PASS Mask Gate

----
## Inrotduction
This repo is official **[PyTorch](https://pytorch.org)** implementation of Patch Automatic Skip Scheme(PASS)(ATC 2022).
You can directly use this module to generate a mask that can automatically choose patches that can be skipped. By adding hardware support, you can achieve higher performance on tiny devices.
## Installation

```shell
pip install pass-mask-gate
```

## Implementation
Our PASS can be implemented in any convolution-based networks by adding a mask gate before convolution modules. 
Here is a simple example for constructing a masked convolution module, some functions needed may not be included:

```python
import torch
import torch.nn as nn
from PASS import MaskGate, RunningMode, MaskMode


class YourModule(nn.Module):
    def __init__(self, conv_inp, conv_oup, kernel_size, current_running_mode):
        """
        An example to construct your own masked convolution module
        :param conv_inp, conv_oup, kernel_size: your convolution parameter
        :param current_running_mode: Pretrain, Fine-tuning or Test 
        """
        super(YourModule, self).__init__()
        self.Gate = MaskGate(patch_size=4, patch_num=1024, input_channel=conv_inp, output_channel=conv_oup,
                             running_mode=current_running_mode)
        self.YourConv = nn.Conv2d(conv_inp, conv_oup, kernel_size)
        self.previous_x = None

    def forward(self, x, current_mask_mode):
        """
        :param current_mask_mode: Positive, Negative or Anchor
        :param x: your input
        """
        if self.previous_x is None:
            if current_mask_mode == MaskMode.Anchor:
                self.previous_x = x.clone()
            return self.conv(x)
        x = self.Gate(x, self.previous_x, current_mask_mode)
        return self.YourConv(x)

    def set_running_mode(self, running_mode):
        """
        Freeze backbone during pretraining, freeze gate during fine-tuning
        """
        if running_mode == RunningMode.GatePreTrain:
            for p in self.parameters():
                p.requires_grad = False
        self.Gate.set_running_mode(running_mode)
```
## Training

### Self-Supervised Pre-Training
Pretrain your own mask gate according to your convolution setting. Here is an example for pretraining:
```python
import torch
import torch.nn as nn
from PASS import MaskGate, RunningMode, MaskMode

def pretrain(your_model, x):
    running_mode = RunningMode.GatePreTrain
    your_model.set_running_mode(running_mode) # freeze your backbone
    pos_x = your_model(x, MaskMode.Positive)
    neg_x = your_model(x, MaskMode.Negative)
    anchor = your_model(x, MaskMode.Anchor)
    loss_function = nn.TripletMarginLoss(margin=200.0, swap=False, reduction='mean')
    loss = loss_function(anchor, pos_x, neg_x)
    loss.backward()
```

### Supervised Fine-Tuning
After pretraining, a supervised fine-tuning is recommended to improve the accuracy. Generally, this process is exact the same as your own backbone training. Here is an example for fine-tuning:
```python
import torch
import torch.nn as nn
from PASS import MaskGate, RunningMode, MaskMode

def fine_tuning(your_model,your_loss_function, x, label):
    running_mode = RunningMode.FineTuning
    your_model.set_running_mode(running_mode) # freeze the gate
    x = your_model(x, MaskMode.Positive)
    loss = your_loss_function(x, label)
    loss.backward()
```
## Test
Positive result is used in inference, here is the example:
```python
import torch
import torch.nn as nn
from PASS import MaskGate, RunningMode, MaskMode

def test(your_model,your_argmax_function, x):
    your_model = your_model.eval()
    x = your_model(x, MaskMode.Positive)
    return your_argmax_function(x)
```
