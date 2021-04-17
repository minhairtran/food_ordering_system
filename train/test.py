import sys
sys.path.append("../../food_ordering_system")
sys.path.append(
    "/home/minhair/Desktop/food_ordering_system/test_pytorch_venv/lib/python3.8/site-packages/")

from comet_ml import Experiment
import numpy as np
import torch.nn.functional as F
import torch.utils.data as data
import torch.nn as nn
import torch
from error_calculating import ErrorCalculating
from text_transform import ConfirmTextTransform
from model import SpeechRecognitionModel

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


def load_data(data):
    mel_spectrogram = data["mel_spectrogram"]
    labels = data["labels"]
    label_lengths = list(map(int, data["label_lengths"].tolist())) 
    input_lengths = list(map(int, data["input_lengths"].tolist())) 

    return mel_spectrogram, labels, input_lengths, label_lengths

def test(model, device, test_loader, criterion, iter_meter, experiment, filename):
    print('\nEvaluating ' + filename + "...")
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

    model = SpeechRecognitionModel(SpeechRecognitionModel.hparams['n_rnn_layers'], SpeechRecognitionModel.hparams['rnn_dim'], \
        6, SpeechRecognitionModel.hparams['n_feats'], SpeechRecognitionModel.hparams['dropout']).to(device)

    try:
        checkpoint = torch.load(SAVED_MODEL_PATH)
        model.load_state_dict(checkpoint)
    except:
        pass
    
    criterion = nn.CTCLoss(blank=0).to(device)

    iter_meter = IterMeter()

    load_data_set = torch.load(DATA_PATH)

    for dataset_index in range(len(load_data_set)):

        mel_spectrogram, labels, input_lengths, label_lengths = load_data(load_data_set[dataset_index])


        # Create test dataset and Dataloader
        test_dataset = Dataset(mel_spectrogram, labels,
                            input_lengths, label_lengths)

        test_loader = data.DataLoader(dataset=test_dataset,
                                    batch_size=SpeechRecognitionModel.hparams["batch_size"],
                                    shuffle=False)

        test(model, device, test_loader, criterion, iter_meter, experiment, dataset_index)
