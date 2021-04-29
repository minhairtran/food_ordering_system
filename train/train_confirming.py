import sys
sys.path.append("../../food_ordering_system")
sys.path.append(
    "/home/minhair/Desktop/food_ordering_system/test_pytorch_venv/lib/python3.8/site-packages/")

from comet_ml import Experiment
from sklearn.model_selection import train_test_split
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torch.nn as nn
import torch
from error_calculating import ErrorCalculating
from text_transform import ConfirmTextTransform
from model import ConfirmingModel

DATA_PATH = "../data/confirming_data/data.pt"
SAVED_MODEL_PATH = "model_confirming.h5"
text_transform = ConfirmTextTransform()
error_calculating = ErrorCalculating()


class Dataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, mel_spectrogram, labels, input_lengths, label_lengths):
        'Initialization'
        self.mel_spectrogram = mel_spectrogram
        self.labels = labels

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.labels)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        mel_spectrogram = self.mel_spectrogram[index]
        labels = self.labels[index]

        return (torch.tensor(mel_spectrogram, dtype=torch.float).detach().requires_grad_(), labels)

class IterMeter(object):
    """keeps track of total iterations"""

    def __init__(self):
        self.val = 0

    def step(self):
        self.val += 1

    def get(self):
        return self.val


def load_data(data):
    mel_spectrogram = data["mel_spectrogram"]
    labels = data["labels"]

    return mel_spectrogram, labels


def train(model, device, train_loader, criterion, optimizer, scheduler, epoch, iter_meter, experiment):
    model.train()
    data_len = len(train_loader.dataset)
    with experiment.train():
        for batch_idx, _data in enumerate(train_loader):
            spectrograms, labels = _data
            spectrograms, labels = spectrograms.to(device), labels.to(
                device)  # spectro (batch, cnn_feature, n_class, time)

            optimizer.zero_grad()

            output = model(spectrograms)  # (batch, time, n_class)
            output = F.log_softmax(output, dim=2)
            output = output.transpose(0, 1)  # (time, batch, n_class)

            loss = criterion(output, labels)
            loss.backward()

            experiment.log_metric('loss', loss.item(), step=iter_meter.get())
            experiment.log_metric(
                'learning_rate', scheduler.get_lr(), step=iter_meter.get())

            optimizer.step()
            scheduler.step()
            iter_meter.step()
            if batch_idx % 100 == 0 or batch_idx == data_len:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(spectrograms), data_len,
                    100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader, criterion, iter_meter, experiment, filename):
    print('\nEvaluating ' + str(filename) + "...")
    model.eval()
    test_loss = 0
    test_cer = []

    with experiment.test():
        with torch.no_grad():
            for i, _data in enumerate(test_loader):
                spectrograms, labels= _data
                spectrograms, labels = spectrograms.to(
                    device), labels.to(device)

                output = model(spectrograms)  # (batch, time, n_class)
                output = F.log_softmax(output, dim=2)
                output = output.transpose(0, 1)  # (time, batch, n_class)

                loss = criterion(output, labels)
                test_loss += loss.item() / len(test_loader)

    experiment.log_metric('test_loss', test_loss, step=iter_meter.get())

    print('Test set: Average loss: {:.4f}\n'.format(
        test_loss))


if __name__ == "__main__":

    # Create an experiment with your api key
    experiment = Experiment(
        api_key="NXte5ivrmxMfwBeYjvahf97PC",
        project_name="general",
        workspace="hai321",
    )

    experiment.add_tags(["food_confirming_data", "deep_speech_model"])
    experiment.set_name("Test confirming data with deepspeech model")

    experiment.log_parameters(ConfirmingModel.hparams)

    # Config gpu/cpu
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(7)
    device = torch.device("cuda" if use_cuda else "cpu")

    model = ConfirmingModel(ConfirmingModel.hparams['n_cnn_layers'], ConfirmingModel.hparams['n_feats'], \
        ConfirmingModel.hparams['stride'], ConfirmingModel.hparams['dropout']).to(device)

    try:
        checkpoint = torch.load(SAVED_MODEL_PATH)
        model.load_state_dict(checkpoint)
    except:
        pass

    # model summaries
    print(model)
    print('Num Model Parameters', sum(
        [param.nelement() for param in model.parameters()]))

    optimizer = optim.AdamW(model.parameters(), ConfirmingModel.hparams["learning_rate"])
    criterion = nn.CrossEntropyLoss().to(device)

    iter_meter = IterMeter()

    load_data_set = torch.load(DATA_PATH)

    for epoch in range(1, ConfirmingModel.hparams["epochs"] + 1):

        for dataset_index in range(len(load_data_set)):
            # Load all data
            mel_spectrogram, labels = load_data(load_data_set[dataset_index])

            # Split into train and test
            mel_spectrogram_train, mel_spectrogram_test, labels_train, labels_test = train_test_split(mel_spectrogram, labels, test_size=ConfirmingModel.hparams['test_size'], shuffle=False)

            # Create train dataset and Dataloader
            train_dataset = Dataset(
                mel_spectrogram_train, labels_train)

            train_loader = data.DataLoader(dataset=train_dataset,
                                        batch_size=ConfirmingModel.hparams["batch_size"],
                                        shuffle=True)

            # Create test dataset and Dataloader
            test_dataset = Dataset(mel_spectrogram_test, labels_test)

            test_loader = data.DataLoader(dataset=test_dataset,
                                        batch_size=ConfirmingModel.hparams["batch_size"],
                                        shuffle=True)

            scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=ConfirmingModel.hparams["learning_rate"],
                                                    steps_per_epoch=int(
                                                        len(train_loader)),
                                                    epochs=ConfirmingModel.hparams["epochs"],
                                                    anneal_strategy='linear')

            train(model, device, train_loader, criterion, optimizer,
                    scheduler, epoch, iter_meter, experiment)

            test(model, device, test_loader, criterion, iter_meter, experiment, dataset_index)
            

    # Save model
    torch.save(model.state_dict(), SAVED_MODEL_PATH)
