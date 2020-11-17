from model.thoughtviz.model import EEG_CLASSIFIER
from dataset.eeg_dataset import EEG_DATASET
from torch.utils.data import DataLoader

# Training config
EPCH = 1000
BS = 16

dataset = EEG_DATASET(data_path='../../dataset/')
dataset.check_info()  # Check the loaded sample len

dat_loader = DataLoader(dataset, shuffle=True, batch_size=BS)

# Model config
model = EEG_CLASSIFIER(eegChanNum=16, sampleLen=dataset.get_SAMPLE_LEN())

for IDX_EPCH in range(EPCH):
    for eeg_sig, label in dat_loader:
        model.train_model(inputs=eeg_sig, targets=label)
