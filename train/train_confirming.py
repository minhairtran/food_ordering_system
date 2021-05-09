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
from model import ConfirmingModel
from text_transform import ConfirmTextTransform

DATA_PATH = "../data/confirming_data/data.pt"
SAVED_MODEL_PATH = "model_confirming.h5"
error_calculating = ErrorCalculating()
text_transform = ConfirmTextTransform()

class GetOutOfLoop(Exception):
    pass

class Dataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, mel_spectrogram, labels, input_lengths, label_lengths):
        'Initialization'
        self.mel_spectrogram = mel_spectrogram
        self.labels = labels
        self.label_lengths = label_lengths
        self.input_lengths = input_lengths

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.labels)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        mel_spectrogram = self.mel_spectrogram[index]
        labels = self.labels[index]
        label_length = self.label_lengths[index]
        input_length = self.input_lengths[index]

        return (mel_spectrogram, labels, input_length, label_length)

def GreedyDecoder(output, labels, label_lengths, blank_label=0, collapse_repeated=True):
    arg_maxes = torch.argmax(output, dim=2)
    print(arg_maxes)

    decodes = []
    targets = []

    for i, args in enumerate(arg_maxes):
        decode = []
        targets.append(text_transform.int_to_text(
            labels[i][:label_lengths[i]].tolist()))
        for j, index in enumerate(args):
            if index != blank_label:
                if collapse_repeated and j != 0 and index == args[j - 1]:
                    continue
                decode.append(index.item())
        decodes.append(text_transform.int_to_text(decode))
    return decodes, targets

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
    label_lengths = list(map(int, data["label_lengths"].tolist())) 
    input_lengths = list(map(int, data["input_lengths"].tolist())) 

    return mel_spectrogram, labels, input_lengths, label_lengths

def train(model, device, train_loader, criterion, optimizer, scheduler, epoch, iter_meter, experiment):
    model.train()
    data_len = len(train_loader.dataset)
    with experiment.train():
        for batch_idx, _data in enumerate(train_loader):
            spectrograms, labels, input_lengths, label_lengths = _data
            spectrograms, labels = spectrograms.to(device), labels.to(
                device)  # spectro (batch, cnn_feature, n_class, time)

            optimizer.zero_grad()

            output = model(spectrograms)  # (batch, time, n_class)
            output = F.log_softmax(output, dim=2)
            output = output.transpose(0, 1)  # (time, batch, n_class)

            loss = criterion(output, labels, input_lengths, label_lengths)
            loss.backward()

            experiment.log_metric('loss', loss.item(), step=iter_meter.get())
            experiment.log_metric(
                'learning_rate', scheduler.get_last_lr(), step=iter_meter.get())

            optimizer.step()
            scheduler.step()
            iter_meter.step()
            if (batch_idx % 5 == 0 and batch_idx != 0) or batch_idx == data_len:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(spectrograms), data_len,
                    100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader, criterion, iter_meter, experiment, filename):
    print('\nEvaluating ' + str(filename) + "...")
    model.eval()
    test_loss = 0
    # test_cer, test_wer = [], []
    test_cer = []

    with experiment.test():
        with torch.no_grad():
            for i, _data in enumerate(test_loader):
                spectrograms, labels, input_lengths, label_lengths = _data
                spectrograms, labels = spectrograms.to(
                    device), labels.to(device)

                output = model(spectrograms)  # (batch, time, n_class)
                output = F.log_softmax(output, dim=2)
                output = output.transpose(0, 1)  # (time, batch, n_class)

                loss = criterion(output, labels, input_lengths, label_lengths)
                test_loss += loss.item() / len(test_loader)

                decoded_preds, decoded_targets = GreedyDecoder(
                    output.transpose(0, 1), labels, label_lengths)

                for j in range(len(decoded_preds)):
                    test_cer.append(error_calculating.cer(decoded_targets[j], decoded_preds[j]))

    avg_cer = sum(test_cer)/len(test_cer)
    experiment.log_metric('test_loss', test_loss, step=iter_meter.get())
    experiment.log_metric('cer', avg_cer, step=iter_meter.get())

    print('Test set: Average loss: {:.4f}, Average CER: {:4f}\n'.format(
        test_loss, avg_cer))

    return avg_cer

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

    model = ConfirmingModel(ConfirmingModel.hparams['n_cnn_layers'], ConfirmingModel.hparams['n_rnn_layers'], ConfirmingModel.hparams['rnn_dim'], \
        ConfirmingModel.hparams['n_class'], ConfirmingModel.hparams['n_feats'], ConfirmingModel.hparams['stride'], ConfirmingModel.hparams['dropout']).to(device)

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
    criterion = nn.CTCLoss(blank=0).to(device)

    iter_meter = IterMeter()

    load_data_set = torch.load(DATA_PATH)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=ConfirmingModel.hparams["learning_rate"],
                                                        steps_per_epoch=66,
                                                        epochs=ConfirmingModel.hparams["epochs"],
                                                        anneal_strategy='linear')

    try:
        for epoch in range(1, ConfirmingModel.hparams["epochs"] + 1):
            epoch_cer = []

            for dataset_index in range(len(load_data_set)):
                # Load all data
                mel_spectrogram, labels, input_lengths, label_lengths = load_data(load_data_set[dataset_index])

            # Split into train and test
                mel_spectrogram_train, mel_spectrogram_test, labels_train, labels_test, input_lengths_train, \
                    input_lengths_test, label_lengths_train, label_lengths_test = train_test_split(mel_spectrogram, labels,
                                                                                            input_lengths, label_lengths, test_size=ConfirmingModel.hparams['test_size'], shuffle=False)

                # Create train dataset and Dataloader
                train_dataset = Dataset(mel_spectrogram_train, labels_train, input_lengths_train, label_lengths_train)

                train_loader = data.DataLoader(dataset=train_dataset,
                                            batch_size=ConfirmingModel.hparams["batch_size"],
                                            shuffle=True if epoch>10 else False)

                print(int(len(train_loader)))
                # Create test dataset and Dataloader
                test_dataset = Dataset(mel_spectrogram_test, labels_test, input_lengths_test, label_lengths_test)

                test_loader = data.DataLoader(dataset=test_dataset,
                                            batch_size=ConfirmingModel.hparams["batch_size"],
                                            shuffle=False)

                train(model, device, train_loader, criterion, optimizer,
                        scheduler, epoch, iter_meter, experiment)

                cer_test = test(model, device, test_loader, criterion, iter_meter, experiment, dataset_index)

                epoch_cer.append(cer_test)

            # Save model
            torch.save(model.state_dict(), SAVED_MODEL_PATH)

            if all(each_cer < 0.001 for each_cer in epoch_cer):
                raise GetOutOfLoop

    except GetOutOfLoop:
        pass