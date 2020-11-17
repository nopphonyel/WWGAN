from torch import nn
from model.thoughtviz.custom_layer import MoGLayer
import torch


class DISCRIM_THOUGHVIZ(nn.Module):
    """
    Currently, this model require a batch size
    """
    __DIS_ADAM_LR = 0.00005
    __DIS_ADAM_BETA_1 = 0.5

    def __init__(self, img_classifier):
        super(DISCRIM_THOUGHVIZ, self).__init__()

        self.img_classifier = img_classifier

        self.conv2dPack = nn.ModuleList()
        self.conv2dPack.append(
            nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=2)
                , nn.LeakyReLU(0.2)
                , nn.Dropout(0.5)
            )
        )
        self.conv2dPack.append(
            nn.Sequential(
                nn.BatchNorm2d(num_features=16)
                , nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1)
                , nn.LeakyReLU(0.2)
                , nn.Dropout(0.5)
            )
        )
        self.conv2dPack.append(
            nn.Sequential(
                nn.BatchNorm2d(num_features=32)
                , nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2)
                , nn.LeakyReLU(0.2)
                , nn.Dropout(0.5)
            )
        )
        self.conv2dPack.append(
            nn.Sequential(
                nn.BatchNorm2d(num_features=64)
                , nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1), nn.LeakyReLU(0.2)
                , nn.LeakyReLU(0.2)
                , nn.Dropout(0.5)
            )
        )
        self.conv2dPack.append(
            nn.Sequential(
                nn.BatchNorm2d(num_features=128)
                , nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2)
                , nn.LeakyReLU(0.2)
                , nn.Dropout(0.5)
            )
        )
        self.conv2dPack.append(
            nn.Sequential(
                nn.BatchNorm2d(num_features=256)
                , nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1)
                , nn.LeakyReLU(0.2)
                , nn.Dropout(0.5)
            )
        )

        # self.flatten = nn.Sequential(
        #    nn.Linear(in_features=''' Do dimension calculation here ''', out_features=1),
        #    nn.Sigmoid()
        # )
        self.sigmoid = nn.Sigmoid()
        self.dense = nn.Linear(in_features=4096, out_features=1)

        self.optim = torch.optim.Adam(self.parameters(), lr=self.__DIS_ADAM_LR, betas=(self.__DIS_ADAM_BETA_1, 0.999))
        self.loss_func = nn.BCELoss()
        self.counter = 0
        self.progress = []

    @staticmethod
    def __find_total_dim(tensor):
        total = 1
        for i in range(1, len(tensor.shape)):
            total = total * tensor.shape[i]
        return total

    """
    Expected tensor to have shape (N,C,H,W,...)
    """

    def forward(self, img):
        for idx, conv2d_layer in enumerate(self.conv2dPack):
            if idx == 0:
                x = conv2d_layer(img)
            else:
                x = conv2d_layer(x)

        flatten_shape = self.__find_total_dim(img)
        x = img.reshape([img.shape[0], flatten_shape])  # flatten operation
        # self.dense = nn.Linear(in_features=x.shape[1], out_features=1)  # Todo: recheck the input_features.

        x = self.dense(x)
        fake = self.sigmoid(x)
        img_class = self.img_classifier(img)
        return fake, img_class

    def train_model(self, inputs, targets):
        outputs = self.forward(inputs)
        loss = self.loss_func(outputs, targets)

        self.counter += 1
        if self.counter % 10 == 0:
            self.progress.append(loss.item())

        self.optim.zero_grad()  # Clear all gradient value before backward (prevent gradient accumulation)
        loss.backward()
        self.optim.step()  # Then step the parameter by grad

    def eval_forward(self, inputs):
        self.eval()
        outputs = self.forward(inputs)
        self.train()
        return outputs

    def export(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path))


class GENERATOR_RGB(nn.Module):
    __GEN_ADAM_LR = 0.00003
    __GEN_ADAM_BETA_1 = 0.5

    def __init__(self, noise_dim, features_dim):
        super(GENERATOR_RGB, self).__init__()
        assert features_dim == noise_dim, "<X>: Currently, noise and features dimension expect to be same."
        self.noise_dim = noise_dim

        # From equation z = MEANi + (STDi * EP) | EP ~ N(0,1)
        self.MoGLayer = MoGLayer(noise_dim=noise_dim)

        self.dense01 = nn.Sequential(
            nn.Linear(in_features=features_dim, out_features=features_dim)
            , nn.Tanh()
        )

        self.batchNorm01 = nn.BatchNorm1d(num_features=features_dim)

        self.dense02 = nn.Sequential(
            nn.Linear(in_features=features_dim, out_features=512 * 4 * 4)
            , nn.ReLU()
        )

        self.conv2dT01 = nn.Sequential(
            # From the paper, they use kern_size = 5, padding = 0
            nn.BatchNorm2d(num_features=512, momentum=0.8)
            , nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, padding=1, stride=2)
            , nn.ReLU()
        )

        self.conv2dT02 = nn.Sequential(
            nn.BatchNorm2d(num_features=256, momentum=0.8)
            , nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, padding=1, stride=2)
            , nn.ReLU()
        )

        self.conv2dT03 = nn.Sequential(
            nn.BatchNorm2d(num_features=128, momentum=0.8)
            , nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, padding=1, stride=2)
            , nn.ReLU()
        )

        self.conv2dT04 = nn.Sequential(
            nn.BatchNorm2d(num_features=64, momentum=0.8)
            , nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=4, padding=1, stride=2)
            , nn.ReLU()
        )

        self.tanh = nn.Tanh()

        self.optim = torch.optim.Adam(self.parameters(), lr=self.__GEN_ADAM_LR, betas=(self.__GEN_ADAM_BETA_1, 0.999))
        self.counter = 0
        self.progress = []

    """
    Expected tensor to have shape (N,C,...)
    """

    def forward(self, eeg_features):
        noise_input = torch.rand(self.noise_dim)
        x = self.MoGLayer(noise_input, noise_input.device.type)
        x = self.dense01(x)
        x = x * eeg_features  # Multiply noise with eeg signal here
        # print(x.shape)  # Expected to be 2 dimension

        # if len(x.shape) >= 2:
        x = self.batchNorm01(x)  # This layer allowed one batch when the model in eval mode.
        x = self.dense02(x)

        x = x.reshape([x.shape[0], 512, 4, 4])
        # x = x.unsqueeze(0) # Try to run the code with out this line first
        x = self.conv2dT01(x)
        x = self.conv2dT02(x)
        x = self.conv2dT03(x)
        x = self.conv2dT04(x)
        x = self.tanh(x)
        return x

    def train_model(self, D, input, targets):
        g_output = self.forward(input)
        d_output = D.forward(g_output)

        loss = D.loss_func(d_output, targets)

        self.counter += 1
        if self.counter % 10 == 0:
            self.progress_loss.append(loss.item())

        self.optim.zero_grad()  # Clear all gradient value before backward (prevent gradient accumulation)
        loss.backward()
        self.optim.step()  # Then step the parameter by grad

    def eval_forward(self, inputs):
        self.eval()
        outputs = self.forward(inputs)
        self.train()
        return outputs

    def export(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path))


class EEG_CLASSIFIER(nn.Module):
    def __init__(self, eegChanNum=14, outputSize=10, sampleLen=32):
        super(EEG_CLASSIFIER, self).__init__()

        self.__DEPENDENT_SIZE = self.calc_dependend_size(sampleLen)

        ## (BATCH, CHAN, EEG_CHAN, EEG_LEN)
        self.batchNorm_1 = nn.BatchNorm2d(num_features=1)
        self.conv2d_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 4))
            , nn.ReLU()
        )

        self.conv2d_2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=25, kernel_size=(eegChanNum, 1))
            , nn.ReLU()
        )

        self.maxPool_1 = nn.MaxPool2d(kernel_size=(1, 3))

        self.conv2d_3 = nn.Sequential(  # Todo: Recheck the data format
            nn.Conv2d(in_channels=1, out_channels=50, kernel_size=(4, 25))
            , nn.ReLU()
        )

        self.maxPool_2 = nn.MaxPool2d(kernel_size=(1, 3))

        self.conv2d_4 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=100, kernel_size=(50, 2))
            , nn.ReLU()
        )

        self.flatten_1 = self.flatten

        self.batchNorm_2 = nn.BatchNorm1d(num_features=self.__DEPENDENT_SIZE)

        self.final_dense_1 = nn.Sequential(
            nn.Linear(in_features=self.__DEPENDENT_SIZE, out_features=100)
            , nn.ReLU()
        )

        self.batchNorm_3 = nn.BatchNorm1d(num_features=100)

        self.final_dense_2 = nn.Sequential(
            nn.Linear(in_features=100, out_features=outputSize)
            , nn.Softmax(dim=1)
        )

        # Some required component
        self.optim = torch.optim.Adam(self.parameters(), lr=self.__LR, weight_decay=self.__DECAY)
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.counter = 0
        self.progress = []

    __LR = 0.0001
    __DECAY = 1e-6

    def calc_dependend_size(self, sample_len):
        return 100

    """
    Expected tensor to have shape (N,C,H,W)
    N : Batch size
    C : Channel
    H : Electrode or EEG Channel (Default is 14)
    W : Sample len (The ThoughtViz use len = 32)
    """

    def forward(self, eeg):
        x = self.batchNorm_1(eeg)
        x = self.conv2d_1(x)
        x = self.conv2d_2(x)
        x = self.maxPool_1(x)

        x = self.channel_first(x)  # This method just swap channel with H and W
        x = self.conv2d_3(x)
        x = self.undo_channel_first(x)
        x = self.maxPool_2(x)
        x = self.conv2d_4(x)
        x = self.flatten(x)
        x = self.batchNorm_2(x)
        eeg_features = self.final_dense_1(x)
        x = self.batchNorm_3(eeg_features)
        eeg_class = self.final_dense_2(x)
        return eeg_features, eeg_class

    def train_model(self, inputs, targets):
        _, outputs = self.forward(inputs)
        loss = self.loss_func(outputs, targets.argmax(1))
        #loss = nn.CrossEntropyLoss(outputs, targets.argmax(1))

        self.counter += 1
        if self.counter % 10 == 0:
            self.progress.append(loss.item())

        self.optim.zero_grad()  # Clear all gradient value before backward (prevent gradient accumulation)
        loss.backward()
        self.optim.step()  # Then step the parameter by grad

    def eval_forward(self, inputs):
        self.eval()
        outputs = self.forward(inputs)
        self.train()
        return outputs

    def export(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path))

    @staticmethod
    def flatten(tensor):
        total = 1
        for i in range(1, len(tensor.shape)):
            total = total * tensor.shape[i]
        return tensor.reshape([tensor.shape[0], total])

    @staticmethod
    def channel_first(tensor):
        assert len(tensor.shape) == 4, "Expect 4D tensor input."
        x = torch.transpose(tensor, 1, 2)
        return torch.transpose(x, 2, 3)

    @staticmethod
    def undo_channel_first(tensor):
        assert len(tensor.shape) == 4, "Expect 4D tensor input."
        x = torch.transpose(tensor, 2, 3)
        return torch.transpose(x, 1, 2)
