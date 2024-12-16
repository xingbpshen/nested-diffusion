import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18, resnet50
from diffusion_utils import make_beta_schedule, p_sample_loop
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import copy


# class ComplexSequenceClassifier(nn.Module):
#     def __init__(self, input_dim, num_classes, num_layers=12, num_heads=5, hidden_dim=10, dropout=0.1):
#         super(ComplexSequenceClassifier, self).__init__()
#         self.input_dim = input_dim
#         self.num_layers = num_layers
#         self.num_heads = num_heads
#         self.hidden_dim = hidden_dim
#         self.dropout = dropout
#
#         # Define the Transformer Encoder layers
#         encoder_layers = TransformerEncoderLayer(d_model=self.input_dim, nhead=self.num_heads,
#                                                  dim_feedforward=self.hidden_dim, dropout=self.dropout)
#         self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=self.num_layers)
#
#         # Define intermediate layers
#         self.intermediate = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout)
#         )
#
#         # Define the final classification layer
#         self.classifier = nn.Linear(hidden_dim, num_classes)
#
#     def forward(self, x):
#         # x shape: [seq_len, batch_size, input_dim]
#         x_trans_seq = self.transformer_encoder(x)
#
#         # Take the representation of the last token
#         last_output = x_trans_seq[-1]
#
#         # Pass through intermediate layers
#         # last_output = self.intermediate(last_output)
#
#         # Classify the sequence based on the last token
#         logits = self.classifier(last_output)
#
#         return logits


class NewClassifier(nn.Module):
    def __init__(self, config):
        super(NewClassifier, self).__init__()
        self.config = config

        self.num_timesteps = config.diffusion.timesteps
        self.granularity = config.diffusion.granularity
        # define a transformer block
        self.mca = nn.Transformer(d_model=10, nhead=2, num_encoder_layers=4)

        self.lin = nn.Sequential(
            nn.Linear(int(config.diffusion.aux_cls.feature_dim), 10),
            nn.ReLU(),)

        self.lin2 = nn.Sequential(
            nn.Linear(int(self.num_timesteps / self.granularity) * 10, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10), )

    def forward(self, selected_time_points, x_feature):
        batch = x_feature.shape[0]
        x_feature_flat = torch.flatten(x_feature, start_dim=1)

        _feature = self.lin(x_feature_flat)
        # add one dimension to make it a sequence
        _feature = _feature.unsqueeze(0)
        logits = self.mca(_feature, selected_time_points)

        # transpose seq and batch dimension
        logits_transposed = logits.transpose(0, 1)  # now shape is (batch, seq, dim)

        # reshape to (batch, seq * dim)
        logits_reshaped = logits_transposed.contiguous().view(batch, -1)  # now shape is (batch, seq * dim)

        logits = self.lin2(logits_reshaped)

        # softmax is included in the loss function
        return logits


class ConditionalLinear(nn.Module):
    def __init__(self, num_in, num_out, n_steps):
        super(ConditionalLinear, self).__init__()
        self.num_out = num_out
        self.lin = nn.Linear(num_in, num_out)
        self.embed = nn.Embedding(n_steps, num_out)
        self.embed.weight.data.uniform_()

    def forward(self, x, t):
        out = self.lin(x)
        gamma = self.embed(t)
        out = gamma.view(-1, self.num_out) * out
        return out


class ConditionalModel(nn.Module):
    def __init__(self, config, guidance=False):
        super(ConditionalModel, self).__init__()
        n_steps = config.diffusion.timesteps + 1
        data_dim = config.model.data_dim
        y_dim = config.data.num_classes
        arch = config.model.arch
        feature_dim = config.model.feature_dim
        hidden_dim = config.model.hidden_dim
        self.guidance = guidance
        # encoder for x
        if config.data.dataset == 'toy':
            self.encoder_x = nn.Linear(data_dim, feature_dim)
        elif config.data.dataset in ['FashionMNIST', 'MNIST', 'CIFAR10', 'CIFAR100', 'IMAGENE100', 'RotatedMNIST',
                                     'ChestXRay', 'ChestXRayAtkFGSM', 'ChestXRayAtkPGD', 'ChestXRayAtkBIM',
                                     'ChestXRayAtkAUTOPGD', 'ChestXRayAtkCW', 'ChestXRayValidate', 'ISICSkinCancer',
                                     'ISICSkinCancerAtkFGSM', 'ISICSkinCancerAtkPGD', 'ISICSkinCancerAtkBIM',
                                     'ISICSkinCancerAtkAUTOPGD', 'ISICSkinCancerAtkCW', 'ISICSkinCancerValidate']:
            if arch == 'linear':
                self.encoder_x = nn.Sequential(
                    nn.Linear(data_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Softplus(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Softplus(),
                    nn.Linear(hidden_dim, feature_dim)
                )
            elif arch == 'simple':
                self.encoder_x = nn.Sequential(
                    nn.Linear(data_dim, 300),
                    nn.BatchNorm1d(300),
                    nn.ReLU(),
                    nn.Linear(300, 100),
                    nn.BatchNorm1d(100),
                    nn.ReLU(),
                    nn.Linear(100, feature_dim)
                )
            elif arch == 'lenet':
                self.encoder_x = LeNet(feature_dim, config.model.n_input_channels, config.model.n_input_padding)
            elif arch == 'lenet5':
                self.encoder_x = LeNet5(feature_dim, config.model.n_input_channels, config.model.n_input_padding)
            else:
                self.encoder_x = FashionCNN(out_dim=feature_dim)
        else:
            self.encoder_x = ResNetEncoder(arch=arch, feature_dim=feature_dim)
        # batch norm layer
        self.norm = nn.BatchNorm1d(feature_dim)

        # Unet
        if self.guidance:
            self.lin1 = ConditionalLinear(y_dim * 2, feature_dim, n_steps)
        else:
            self.lin1 = ConditionalLinear(y_dim, feature_dim, n_steps)
        self.unetnorm1 = nn.BatchNorm1d(feature_dim)
        self.lin2 = ConditionalLinear(feature_dim, feature_dim, n_steps)
        self.unetnorm2 = nn.BatchNorm1d(feature_dim)
        self.lin3 = ConditionalLinear(feature_dim, feature_dim, n_steps)
        self.unetnorm3 = nn.BatchNorm1d(feature_dim)
        self.lin4 = nn.Linear(feature_dim, y_dim)

    def forward(self, x, y, t, yhat=None):
        x = self.encoder_x(x)
        x = self.norm(x)
        if self.guidance:
            y = torch.cat([y, yhat], dim=-1)
        y = self.lin1(y, t)
        y = self.unetnorm1(y)
        y = F.softplus(y)
        y = x * y
        y = self.lin2(y, t)
        y = self.unetnorm2(y)
        y = F.softplus(y)
        y = self.lin3(y, t)
        y = self.unetnorm3(y)
        y = F.softplus(y)
        return self.lin4(y)


# class NewConditionalModel(nn.Module):
#     def __init__(self, config, guidance=False):
#         super(NewConditionalModel, self).__init__()
#         self.conditional_model = ConditionalModel(config=config, guidance=guidance)
#         self.classifier = NewClassifier(config=config)
#
#     def forward(self, x, y, t, yhat=None,
#                 include_classifier=False, cond_pred_model=None, x_unflat=None):
#         if yhat is None:
#             raise ValueError('yhat is None')
#         x_clone = x.clone().detach()
#         yhat_clone = yhat.clone().detach()
#
#         noise_estimation = self.conditional_model(x, y, t, yhat)
#
#         if include_classifier:
#             classification_logits = self.classifier(x_clone, yhat_clone,
#                                                     copy.deepcopy(self.conditional_model),
#                                                     cond_pred_model, x_unflat)
#         else:
#             classification_logits = torch.zeros_like(yhat_clone)
#
#         return noise_estimation, classification_logits


# Simple convnet
# ---------------------------------------------------------------------------------
# Revised from: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
# ---------------------------------------------------------------------------------
class SimNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        return x


# FashionCNN
# --------------------------------------------------------------------------------------------------
# Revised from: https://www.kaggle.com/code/pankajj/fashion-mnist-with-pytorch-93-accuracy/notebook
# --------------------------------------------------------------------------------------------------
class FashionCNN(nn.Module):

    def __init__(self, out_dim=10, use_for_guidance=False):
        super(FashionCNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.use_for_guidance = use_for_guidance
        if self.use_for_guidance:
            self.fc1 = nn.Linear(in_features=64 * 6 * 6, out_features=600)
            self.drop = nn.Dropout2d(0.25)
            self.fc2 = nn.Linear(in_features=600, out_features=120)
            self.fc3 = nn.Linear(in_features=120, out_features=out_dim)
        else:
            self.fc1 = nn.Linear(in_features=64 * 6 * 6, out_features=out_dim)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        if self.use_for_guidance:
            out = self.drop(out)
            out = self.fc2(out)
            out = self.fc3(out)

        return out


# ResNet 18 or 50 as image encoder
class ResNetEncoder(nn.Module):
    def __init__(self, arch='resnet18', feature_dim=128):
        super(ResNetEncoder, self).__init__()

        self.f = []
        if arch == 'resnet50':
            backbone = resnet50()
        elif arch == 'resnet18':
            backbone = resnet18()
        for name, module in backbone.named_children():
            if not isinstance(module, nn.Linear):
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        self.featdim = backbone.fc.weight.shape[1]
        self.g = nn.Linear(self.featdim, feature_dim)

    def forward_feature(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        feature = self.g(feature)
        return feature

    def forward(self, x):
        feature = self.forward_feature(x)
        return feature


# LeNet
class LeNet(nn.Module):
    def __init__(self, num_classes=10, n_input_channels=1, n_input_padding=2):
        super(LeNet, self).__init__()
        # CIFAR-10 with shape (3, 32, 32): n_input_channels=3, n_input_padding=0
        # FashionMNIST and MNIST with shape (1, 28, 28): n_input_channels=1, n_input_padding=2
        self.conv1 = nn.Conv2d(in_channels=n_input_channels, out_channels=6,
                               kernel_size=5, stride=1, padding=n_input_padding)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16,
                               kernel_size=5, stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120,
                               kernel_size=5, stride=1, padding=0)
        self.linear1 = nn.Linear(120, 84)
        self.linear2 = nn.Linear(84, num_classes)
        self.tanh = nn.Tanh()
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.tanh(x)
        x = self.avgpool(x)
        x = self.conv2(x)
        x = self.tanh(x)
        x = self.avgpool(x)
        x = self.conv3(x)
        x = self.tanh(x)
        x = x.reshape(x.shape[0], -1)
        x = self.linear1(x)
        x = self.tanh(x)
        x = self.linear2(x)
        return x


class LeNet5(nn.Module):
    def __init__(self, num_classes=10, n_input_channels=1, n_input_padding=2):
        super(LeNet5, self).__init__()
        # CIFAR-10 with shape (3, 32, 32): n_input_channels=3, n_input_padding=0
        # FashionMNIST and MNIST with shape (1, 28, 28): n_input_channels=1, n_input_padding=2
        self.layer1 = nn.Sequential(
            nn.Conv2d(n_input_channels, 6, kernel_size=5, stride=1, padding=n_input_padding),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2))  # apply average pooling instead
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2))
        self.fc0 = nn.Linear(400, 120)
        # self.fc0 = nn.Linear(256, 120)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(84, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc0(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out
