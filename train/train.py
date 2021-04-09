import sys
sys.path.append("../../food_ordering_system")
sys.path.append(
    "/home/minhair/Desktop/food_ordering_system/test_pytorch_venv/lib/python3.8/site-packages/")

from comet_ml import Experiment
from food_ordering_system.train.TextTransform import TextTransform
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torch.nn.functional as F
import torchaudio
import numpy as np
from sklearn.model_selection import train_test_split
import json
import os


DATASET_PATH = "../../food_ordering_system/confirming_dataset"
JSON_PATH = "../../food_ordering_system/data/confirming_data/data.json"
LEARNING_RATE = 5e-4
BATCH_SIZE = 5
EPOCHS = 10
N_CNN_LAYERS = 3
N_RNN_LAYERS = 5
RNN_DIM = 512
N_CLASS_CONFIRM_DATA = 6
N_FEATS = 128
STRIDE = 2
DROPOUT = 0.1
TEST_SIZE = 0.2


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

def _levenshtein_distance(ref, hyp):
    """Levenshtein distance is a string metric for measuring the difference
    between two sequences. Informally, the levenshtein disctance is defined as
    the minimum number of single-character edits (substitutions, insertions or
    deletions) required to change one word into the other. We can naturally
    extend the edits to word level when calculate levenshtein disctance for
    two sentences.
    """
    m = len(ref)
    n = len(hyp)

    # special case
    if ref == hyp:
        return 0
    if m == 0:
        return n
    if n == 0:
        return m

    if m < n:
        ref, hyp = hyp, ref
        m, n = n, m

    # use O(min(m, n)) space
    distance = np.zeros((2, n + 1), dtype=np.int32)

    # initialize distance matrix
    for j in range(0, n + 1):
        distance[0][j] = j

    # calculate levenshtein distance
    for i in range(1, m + 1):
        prev_row_idx = (i - 1) % 2
        cur_row_idx = i % 2
        distance[cur_row_idx][0] = i
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                distance[cur_row_idx][j] = distance[prev_row_idx][j - 1]
            else:
                s_num = distance[prev_row_idx][j - 1] + 1
                i_num = distance[cur_row_idx][j - 1] + 1
                d_num = distance[prev_row_idx][j] + 1
                distance[cur_row_idx][j] = min(s_num, i_num, d_num)

    return distance[m % 2][n]

def char_errors(reference, hypothesis, ignore_case=False, remove_space=False):
    """Compute the levenshtein distance between reference sequence and
    hypothesis sequence in char-level.
    :param reference: The reference sentence.
    :type reference: basestring
    :param hypothesis: The hypothesis sentence.
    :type hypothesis: basestring
    :param ignore_case: Whether case-sensitive or not.
    :type ignore_case: bool
    :param remove_space: Whether remove internal space characters
    :type remove_space: bool
    :return: Levenshtein distance and length of reference sentence.
    :rtype: list
    """
    if ignore_case == True:
        reference = reference.lower()
        hypothesis = hypothesis.lower()

    join_char = ' '
    if remove_space == True:
        join_char = ''

    reference = join_char.join(filter(None, reference.split(' ')))
    hypothesis = join_char.join(filter(None, hypothesis.split(' ')))

    edit_distance = _levenshtein_distance(reference, hypothesis)
    return float(edit_distance), len(reference)

def cer(reference, hypothesis, ignore_case=False, remove_space=False):
    """Calculate charactor error rate (CER). CER compares reference text and
    hypothesis text in char-level. CER is defined as:
    .. math::
        CER = (Sc + Dc + Ic) / Nc
    where
    .. code-block:: text
        Sc is the number of characters substituted,
        Dc is the number of characters deleted,
        Ic is the number of characters inserted
        Nc is the number of characters in the reference
    We can use levenshtein distance to calculate CER. Chinese input should be
    encoded to unicode. Please draw an attention that the leading and tailing
    space characters will be truncated and multiple consecutive space
    characters in a sentence will be replaced by one space character.
    :param reference: The reference sentence.
    :type reference: basestring
    :param hypothesis: The hypothesis sentence.
    :type hypothesis: basestring
    :param ignore_case: Whether case-sensitive or not.
    :type ignore_case: bool
    :param remove_space: Whether remove internal space characters
    :type remove_space: bool
    :return: Character error rate.
    :rtype: float
    :raises ValueError: If the reference length is zero.
    """
    edit_distance, ref_len = char_errors(reference, hypothesis, ignore_case,
                                         remove_space)

    if ref_len == 0:
        raise ValueError("Length of reference should be greater than 0.")

    cer = float(edit_distance) / ref_len
    return cer


text_transform = TextTransform()


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


class CNNLayerNorm(nn.Module):
    """Layer normalization built for cnns input"""

    def __init__(self, n_feats):
        super(CNNLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(n_feats)

    def forward(self, x):
        x = x.transpose(2, 3).contiguous()  # (batch, channel, time, feature)
        x = self.layer_norm(x)
        return x.transpose(2, 3).contiguous() # (batch, channel, feature, time)


class ResidualCNN(nn.Module):
    """Residual CNN inspired by https://arxiv.org/pdf/1603.05027.pdf
        except with layer norm instead of batch norm
    """

    def __init__(self, in_channels, out_channels, kernel, stride, dropout, n_feats):
        super(ResidualCNN, self).__init__()

        self.cnn1 = nn.Conv2d(in_channels, out_channels,
                              kernel, stride, padding=kernel//2)
        self.cnn2 = nn.Conv2d(out_channels, out_channels,
                              kernel, stride, padding=kernel//2)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm1 = CNNLayerNorm(n_feats)
        self.layer_norm2 = CNNLayerNorm(n_feats)

    def forward(self, x):
        residual = x  # (batch, channel, feature, time)
        x = self.layer_norm1(x)
        x = F.gelu(x)
        x = self.dropout1(x)
        x = self.cnn1(x)
        x = self.layer_norm2(x)
        x = F.gelu(x)
        x = self.dropout2(x)
        x = self.cnn2(x)
        x += residual
        return x  # (batch, channel, feature, time)


class BidirectionalGRU(nn.Module):

    def __init__(self, rnn_dim, hidden_size, dropout, batch_first):
        super(BidirectionalGRU, self).__init__()

        self.BiGRU = nn.GRU(
            input_size=rnn_dim, hidden_size=hidden_size,
            num_layers=1, batch_first=batch_first, bidirectional=True)
        self.layer_norm = nn.LayerNorm(rnn_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        x = F.gelu(x)
        x, _ = self.BiGRU(x)
        x = self.dropout(x)
        return x


class SpeechRecognitionModel(nn.Module):

    def __init__(self, n_cnn_layers, n_rnn_layers, rnn_dim, n_class, n_feats, stride=2, dropout=0.1):
        super(SpeechRecognitionModel, self).__init__()
        n_feats = n_feats//2
        # cnn for extracting heirachal features
        self.cnn = nn.Conv2d(1, 32, 3, stride=stride, padding=3//2)

        # n residual cnn layers with filter size of 32
        self.rescnn_layers = nn.Sequential(*[
            ResidualCNN(32, 32, kernel=3, stride=1,
                        dropout=dropout, n_feats=n_feats)
            for _ in range(n_cnn_layers)
        ])
        self.fully_connected = nn.Linear(n_feats*32, rnn_dim)
        self.birnn_layers = nn.Sequential(*[
            BidirectionalGRU(rnn_dim=rnn_dim if i == 0 else rnn_dim*2,
                             hidden_size=rnn_dim, dropout=dropout, batch_first=i == 0)
            for i in range(n_rnn_layers)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(rnn_dim*2, rnn_dim),  # birnn returns rnn_dim*2
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_dim, n_class)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.rescnn_layers(x)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2],
                   sizes[3])  # (batch, feature, time)
        x = x.transpose(1, 2)  # (batch, time, feature)
        x = self.fully_connected(x)
        x = self.birnn_layers(x)
        x = self.classifier(x)
        return x


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

    mel_spectrogram = np.array(data["mel_spectrogram"], dtype=object)
    labels = np.array(data["labels"], dtype=object)
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


def tensorize(mel_spectrogram_not_tensorized, labels_not_tensorized):
    mel_spectrogram, labels = [], []

    for spectrogram in mel_spectrogram_not_tensorized:
        mel_spectrogram.append(torch.Tensor(spectrogram))

    for label in labels_not_tensorized:
        labels.append(torch.Tensor(label))

    mel_spectrogram = nn.utils.rnn.pad_sequence(
        mel_spectrogram, batch_first=True).unsqueeze(1).transpose(2, 3)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

    return mel_spectrogram, labels


if __name__ == "__main__":

    # Create an experiment with your api key
    experiment = Experiment(
        api_key="NXte5ivrmxMfwBeYjvahf97PC",
        project_name="general",
        workspace="hai321",
    )

    experiment.add_tags(["confirm_data", "deep_speech_model"])
    experiment.set_name("Test confirm data with deepspeech model")

    hparams = {
        "n_cnn_layers": N_CNN_LAYERS,
        "n_rnn_layers": N_RNN_LAYERS,
        "rnn_dim": RNN_DIM,
        "n_class": N_CLASS_CONFIRM_DATA,
        "n_feats": N_FEATS,
        "stride": STRIDE,
        "dropout": DROPOUT,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS
    }

    experiment.log_parameters(hparams)

    # Config gpu/cpu
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(7)
    device = torch.device("cuda" if use_cuda else "cpu")

    # Load all data
    mel_spectrogram_not_tensorized, labels_not_tensorized, input_lengths, label_lengths = load_data(
        DATA_PATH)

    # Tensorize data
    mel_spectrogram, labels = tensorize(
        mel_spectrogram_not_tensorized, labels_not_tensorized)

    # Split into train and test
    mel_spectrogram_train, mel_spectrogram_test, labels_train, labels_test, input_lengths_train, \
        input_lengths_test, label_lengths_train, label_lengths_test, = train_test_split(mel_spectrogram, labels,
                                                                                        input_lengths, label_lengths, test_size=TEST_SIZE, shuffle=True)

    # Create train dataset and Dataloader
    train_dataset = Dataset(
        mel_spectrogram_train, labels_train, input_lengths_train, label_lengths_train)

    train_loader = data.DataLoader(dataset=train_dataset,
                                   batch_size=BATCH_SIZE,
                                   shuffle=False)

    del input_lengths_train, label_lengths_train, mel_spectrogram_train, labels_train, train_dataset

    # Create test dataset and Dataloader
    test_dataset = Dataset(mel_spectrogram_test, labels_test,
                           input_lengths_test, label_lengths_test)

    test_loader = data.DataLoader(dataset=test_dataset,
                                  batch_size=BATCH_SIZE,
                                  shuffle=False)

    del input_lengths_test, label_lengths_test, mel_spectrogram_test, labels_test, test_dataset

    model = SpeechRecognitionModel(
        N_CNN_LAYERS, N_RNN_LAYERS, RNN_DIM, N_CLASS_CONFIRM_DATA, N_FEATS, STRIDE, DROPOUT).to(device)

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

        del train_loader

        test(model, device, test_loader, criterion,
             epoch, iter_meter, experiment)

        del test_loader

        if (epoch != EPOCHS):

            # Shuffle test and train after each epoch
            # Split into train and test
            mel_spectrogram_train, mel_spectrogram_test, labels_train, labels_test, input_lengths_train, \
                input_lengths_test, label_lengths_train, label_lengths_test, = train_test_split(mel_spectrogram, labels,
                                                                                                input_lengths, label_lengths, test_size=TEST_SIZE, shuffle=True)
        
            # Create train dataset and Dataloader
            train_dataset = Dataset(
                mel_spectrogram_train, labels_train, input_lengths_train, label_lengths_train)

            train_loader = data.DataLoader(dataset=train_dataset,
                                           batch_size=BATCH_SIZE,
                                           shuffle=False)

            del input_lengths_train, label_lengths_train, mel_spectrogram_train, labels_train, train_dataset


            # Create test dataset and Dataloader
            test_dataset = Dataset(mel_spectrogram_test, labels_test,
                                   input_lengths_test, label_lengths_test)

            test_loader = data.DataLoader(dataset=test_dataset,
                                          batch_size=BATCH_SIZE,
                                          shuffle=False)

            del input_lengths_test, label_lengths_test, mel_spectrogram_test, labels_test, test_dataset


    # Save model
    torch.save(model.state_dict(), SAVED_MODEL_PATH)
