import torch
from torch import nn

from model.semi_supervised.layer import MoGLayer
# from layer import MoGLayer


class SemanticImageExtractor(nn.Module):
    """
    This class expected image as input with size (224x224x3)
    """

    def __init__(self, output_class_num, feature_size=200):
        super(SemanticImageExtractor, self).__init__()
        self.alx_layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.alx_layer2 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.alx_layer3 = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.alx_layer4 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.alx_layer5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((6, 6))
        # return the same number of features but change width and height of img

        self.fc06 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU()
        )

        self.fc07 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(4096, feature_size),
            nn.ReLU()
        )

        self.fc08 = nn.Sequential(
            nn.Linear(feature_size, output_class_num),
            nn.Softmax())

    def forward(self, img):
        x = self.alx_layer1(img)
        x = self.alx_layer2(x)
        x = self.alx_layer3(x)
        x = self.alx_layer4(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc06(x)
        semantic_features = self.fc07(x).unsqueeze(1)
        p_label = self.fc08(semantic_features)
        return semantic_features, p_label


class SemanticEEGExtractor(nn.Module):
    def __init__(self, expected_shape: torch.Tensor, output_class_num: int, feature_size=200):
        """
        expected_shape [Batch_size, eeg_features, eeg_channel, sample_len]
        """
        super(SemanticEEGExtractor, self).__init__()

        self.batch_norm = nn.BatchNorm1d(num_features=expected_shape.shape[1])

        self.fc01 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(expected_shape.shape[1] * expected_shape.shape[2], 4096),
            nn.LeakyReLU()
        )

        self.fc02 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(4096, feature_size),
            nn.LeakyReLU()
        )

        self.fc03 = nn.Sequential(
            nn.Linear(feature_size, output_class_num),
            nn.Softmax())

    def forward(self, eeg: torch.Tensor):
        x = self.batch_norm(eeg)
        x = x.reshape([x.shape[0], -1])
        x = self.fc01(x)
        semantic_features = self.fc02(x).unsqueeze(1)
        label = self.fc03(semantic_features)
        return semantic_features, label


class Generator(nn.Module):  # <<- CGAN
    # How can we input both label and features?
    def __init__(self, sem_size):
        """
        This model will use the same arch as ThoughtViz
        :param sem_size: Semantic features size
        """
        super(Generator, self).__init__()
        self.dense01 = nn.Sequential(
            nn.Linear(sem_size, sem_size),
            nn.LeakyReLU()
        )

        self.MoGLayer = MoGLayer(noise_dim=sem_size)

        self.dense02 = nn.Sequential(
            nn.BatchNorm1d(sem_size + 10),
            nn.Linear(sem_size + 10, 512 * 4 * 4),
            nn.LeakyReLU()
        )

        self.deconv1 = nn.Sequential(
            nn.BatchNorm2d(512, 0.8),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU()
        )
        self.deconv2 = nn.Sequential(
            nn.BatchNorm2d(256, 0.8),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU()
        )
        self.deconv3 = nn.Sequential(
            nn.BatchNorm2d(128, 0.8),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU()
        )
        self.deconv4 = nn.Sequential(
            nn.BatchNorm2d(64, 0.8),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Tanh()
        )

    @staticmethod
    def __forward_check(z, eeg_semantic, eeg_label):
        if eeg_semantic.shape[1] != z.shape[1]:
            raise RuntimeError("\'EEG Semantic\' and noise expected to be the same size")
        if eeg_label.shape[1] != 10:
            raise RuntimeError("Incorrect shape of vector \'eeg_label\'")

    # -- Expected shape --
    # z.shape = (2094,)
    # eeg_semantic.shape = (200,)
    # label.shape = (10,)
    def forward(self, z, semantic, label):
        # First, we need to concat.
        # Problem
        #   Should we concat and deconvolution it?
        #   Second problem, what is the size of z
        self.__forward_check(z, semantic, label)
        x = self.dense01(z)
        x = x * semantic
        x = torch.cat((x, label), dim=1)
        x = self.dense02(x)
        x = x.reshape([x.shape[0], 512, 4, 4])

        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        return x


# Should D1 and D2 takes an real/gen image as an input?
# D1 : Image only
# D2 : Semantic features and label
class D1(nn.Module):
    def __init__(self):
        super(D1, self).__init__()
        self.conv1 = nn.Sequential(  # Currently we input black and white img
            nn.BatchNorm2d(num_features=2),
            nn.Conv2d(2, 64, kernel_size=5, stride=2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5)
        )
        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(num_features=64),
            nn.Conv2d(64, 128, kernel_size=5, stride=2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5)
        )
        self.conv3 = nn.Sequential(
            nn.BatchNorm2d(num_features=128),
            nn.Conv2d(128, 256, kernel_size=5, stride=2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5)
        )
        self.conv4 = nn.Sequential(
            nn.BatchNorm2d(num_features=256),
            nn.Conv2d(256, 512, kernel_size=5, stride=2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5)
        )
        self.final_fc = nn.Sequential(
            nn.Linear(in_features=512, out_features=1),
            nn.Sigmoid()
        )

    @staticmethod
    def __forward_check(x):
        shape = (x.shape[1], x.shape[2], x.shape[3])
        if shape != (1, 64, 64):
            raise RuntimeError("Expected shape", (1, 64, 64))

    def forward(self, x1, x2):
        '''

        :param x1: First image to input
        :param x2: Second image to input but... How we gonna concat? concat in channel dim? YES I THINK WE CAN!
        :return: real or not
        '''
        self.__forward_check(x1)
        self.__forward_check(x2)
        x = torch.cat((x1, x2), dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.reshape((-1, 512))
        x = self.final_fc(x)
        return x


class D2(nn.Module):
    def __init__(self):
        super(D2, self).__init__()

        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(num_features=1),
            nn.Conv2d(1, 64, kernel_size=5, stride=2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5)
        )
        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(num_features=64),
            nn.Conv2d(64, 128, kernel_size=5, stride=2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5)
        )
        self.conv3 = nn.Sequential(
            nn.BatchNorm2d(num_features=128),
            nn.Conv2d(128, 256, kernel_size=5, stride=2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5)
        )
        self.conv4 = nn.Sequential(
            nn.BatchNorm2d(num_features=256),
            nn.Conv2d(256, 512, kernel_size=5, stride=2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5)
        )

        self.final_fc = nn.Sequential(
            nn.Linear(722, 54),
            nn.LeakyReLU(0.2),
            nn.Linear(54, 1),
            nn.Sigmoid()
        )

    @staticmethod
    def __forward_check(img, eeg_features, eeg_label):
        if eeg_features.shape[1] != 200:
            raise RuntimeError("Expected features size = 210")
        if eeg_label.shape[1] != 10:
            raise RuntimeError("Expected shape size = 10")
        img_shape = (img.shape[1], img.shape[2], img.shape[3])
        if img_shape != (1, 64, 64):
            raise RuntimeError("Expected shape", (1, 64, 64))

    def forward(self, img, eeg_features, eeg_label):
        self.__forward_check(img, eeg_features, eeg_label)
        x = self.conv1(img)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = x.reshape((-1, 512))
        x = torch.cat((x, eeg_features), 1)  # Concat eeg_features
        x = torch.cat((x, eeg_label), 1)  # Concat label
        x = self.final_fc(x)
        return x


def __test_execution():
    eeg_test = torch.rand((12, 16, 58))
    eeg_features_model = SemanticEEGExtractor(expected_shape=eeg_test, output_class_num=10)
    semantic_features, label = eeg_features_model(eeg_test)

    semantic_features = semantic_features.squeeze(1)
    label = label.squeeze(1)

    noise = torch.rand((12, semantic_features.shape[1]))
    gen = Generator(semantic_features.shape[1])
    out = gen.forward(z=noise, semantic=semantic_features, label=label)
    d1_test = D1()
    d2_test = D2()
    print(out.shape)
    print(d1_test.forward(out, out).shape)
    print(d2_test.forward(out, semantic_features, label).shape)

#__test_execution()
