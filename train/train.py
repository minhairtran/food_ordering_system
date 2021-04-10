import sys
sys.path.append("../../food_ordering_system")
sys.path.append(
    "/home/minhair/Desktop/food_ordering_system/test_pytorch_venv/lib/python3.8/site-packages/")

from comet_ml import Experiment
import os
import json
from sklearn.model_selection import train_test_split
import numpy as np
import torchaudio
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torch.nn as nn
import torch
from error_calculating import ErrorCalculating
from text_transform import ConfirmTextTransform
from model import SpeechRecognitionModel

DATA_PATH = "../data/confirming_data/data.json"
SAVED_MODEL_PATH = "model_confirming.h5"
text_transform = ConfirmTextTransform()


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

        return (torch.tensor(mel_spectrogram, dtype=torch.float).detach().requires_grad_(), labels, input_length, label_length)


def GreedyDecoder(output, labels, label_lengths, blank_label=0, collapse_repeated=True):
    arg_maxes = torch.argmax(output, dim=2)
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


def load_data(data_path):
    """Loads training dataset from json file.
        :param data_path (str): Path to json file containing data
        :return X (ndarray): Inputs
        :return y (ndarray): Targets
    """

    mel_spectrogram = []
    labels = []

    with open(data_path, "r") as fp:
        data = json.load(fp)

    mel_spectrogram = data["mel_spectrogram"]
    labels = data["labels"]
    label_lengths = np.array(data["label_lengths"])
    input_lengths = np.array(data["input_lengths"])

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
                'learning_rate', scheduler.get_lr(), step=iter_meter.get())

            optimizer.step()
            scheduler.step()
            iter_meter.step()
            if batch_idx % 100 == 0 or batch_idx == data_len:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(spectrograms), data_len,
                    100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader, criterion, epoch, iter_meter, experiment):
    print('\nevaluating...')
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
                    test_cer.append(cer(decoded_targets[j], decoded_preds[j]))

    avg_cer = sum(test_cer)/len(test_cer)
    experiment.log_metric('test_loss', test_loss, step=iter_meter.get())
    experiment.log_metric('cer', avg_cer, step=iter_meter.get())

    print('Test set: Average loss: {:.4f}, Average CER: {:4f}\n'.format(
        test_loss, avg_cer))


if __name__ == "__main__":

    # Create an experiment with your api key
    experiment = Experiment(
        api_key="NXte5ivrmxMfwBeYjvahf97PC",
        project_name="general",
        workspace="hai321",
    )

    experiment.add_tags(["confirm_data", "deep_speech_model"])
    experiment.set_name("Test confirm data with deepspeech model")

    experiment.log_parameters(SpeechRecognitionModel.hparams)

    # Config gpu/cpu
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(7)
    device = torch.device("cuda" if use_cuda else "cpu")

    # Load all data
    mel_spectrogram, labels, input_lengths, label_lengths = load_data(
        DATA_PATH)

    # Split into train and test
    mel_spectrogram_train, mel_spectrogram_test, labels_train, labels_test, input_lengths_train, \
        input_lengths_test, label_lengths_train, label_lengths_test = train_test_split(mel_spectrogram, labels,
                                                                                       input_lengths, label_lengths, test_size=hparams['test_size'], shuffle=True)

    # Create train dataset and Dataloader
    train_dataset = Dataset(
        mel_spectrogram_train, labels_train, input_lengths_train, label_lengths_train)

    train_loader = data.DataLoader(dataset=train_dataset,
                                   batch_size=BATCH_SIZE,
                                   shuffle=False)

    # Create test dataset and Dataloader
    test_dataset = Dataset(mel_spectrogram_test, labels_test,
                           input_lengths_test, label_lengths_test)

    test_loader = data.DataLoader(dataset=test_dataset,
                                  batch_size=BATCH_SIZE,
                                  shuffle=False)

    model = SpeechRecognitionModel(SpeechRecognitionModel.hparams['n_cnn_layers'], SpeechRecognitionModel.hparams['n_rnn_layers'], SpeechRecognitionModel.hparams['rnn_dim'],
                                   SpeechRecognitionModel.hparams['n_class'], SpeechRecognitionModel.hparams['n_feats'], SpeechRecognitionModel.hparams['stride'], SpeechRecognitionModel.hparams['dropout']).to(device)

    # model summaries
    print(model)
    print('Num Model Parameters', sum(
        [param.nelement() for param in model.parameters()]))

    optimizer = optim.AdamW(model.parameters(), LEARNING_RATE)
    criterion = nn.CTCLoss(blank=0).to(device)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LEARNING_RATE,
                                              steps_per_epoch=int(
                                                  len(train_loader)),
                                              epochs=EPOCHS,
                                              anneal_strategy='linear')

    iter_meter = IterMeter()

    for epoch in range(1, EPOCHS + 1):
        train(model, device, train_loader, criterion, optimizer,
              scheduler, epoch, iter_meter, experiment)

        test(model, device, test_loader, criterion,
             epoch, iter_meter, experiment)

    # Save model
    torch.save(model.state_dict(), SAVED_MODEL_PATH)
