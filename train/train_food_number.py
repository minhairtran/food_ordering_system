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
from model import FoodNumberModel
from sklearn.metrics import precision_score


DATA_PATH = ["../data/food_number_data/data_set_0.pt", "../data/food_number_data/data_set_1.pt", "../data/food_number_data/data_set_2.pt",
             "../data/food_number_data/data_set_3.pt", "../data/food_number_data/data_set_4.pt", "../data/food_number_data/data_set_5.pt",
             "../data/food_number_data/data_set_6.pt", "../data/food_number_data/data_set_7.pt", "../data/food_number_data/data_set_8.pt",
             "../data/food_number_data/data_set_9.pt", "../data/food_number_data/data_set_10.pt"]
SAVED_MODEL_PATH = "model_food_number.h5"
error_calculating = ErrorCalculating()

class GetOutOfLoop( Exception ):
    pass

class Dataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, mel_spectrogram, labels):
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

        return (mel_spectrogram, labels)


class IterMeter(object):
    """keeps track of total iterations"""

    def __init__(self):
        self.val = 0

    def step(self):
        self.val += 1

    def get(self):
        return self.val


def load_data(data_path):
    data = torch.load(data_path)
    mel_spectrogram = data["mel_spectrogram"]
    labels = data["labels"]

    return mel_spectrogram, labels

def decoder(output):
    arg_maxes = torch.argmax(output, dim=1).tolist()
    return arg_maxes

def train(model, device, train_loader, criterion, optimizer, scheduler, epoch, iter_meter, experiment):
    model.train()
    data_len = len(train_loader.dataset)
    train_precision_average = []
    with experiment.train():
        for batch_idx, _data in enumerate(train_loader):
            spectrograms, labels = _data
            spectrograms, labels = spectrograms.to(device), labels.to(
                device)  # spectro (batch, cnn_feature, n_class, time)

            optimizer.zero_grad()

            output = model(spectrograms)  # (batch, time, n_class)
            output = F.log_softmax(output, dim=1)
            # output = output.transpose(0, 1)  # (time, batch, n_class)

            loss = criterion(output, labels)
            loss.backward()

            experiment.log_metric('loss', loss.item(), step=iter_meter.get())
            experiment.log_metric(
                'learning_rate', scheduler.get_last_lr(), step=iter_meter.get())
            
            label_pred = decoder(output)

            train_precision = precision_score(np.array(label_pred), np.array(labels.tolist()), average='micro')
            train_precision_average.append(train_precision)

            optimizer.step()
            scheduler.step()
            iter_meter.step()
            if (batch_idx % 100 == 0 and batch_idx != 0) or batch_idx == data_len:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tPrecision: {:.2f}%'.format(
                    epoch, batch_idx * len(spectrograms), data_len,
                    100. * batch_idx / len(train_loader), loss.item(), 100*np.mean(train_precision_average)))


def test(model, device, test_loader, criterion, iter_meter, experiment, filename):
    print('\nEvaluating ' + str(filename) + "...")
    model.eval()
    test_loss = 0
    test_cer = []

    test_precision_average = []

    with experiment.test():
        with torch.no_grad():
            for i, _data in enumerate(test_loader):
                spectrograms, labels= _data
                spectrograms, labels = spectrograms.to(
                    device), labels.to(device)

                output = model(spectrograms)  # (batch, time, n_class)
                output = F.log_softmax(output, dim=1)

                loss = criterion(output, labels)
                test_loss += loss.item() / len(test_loader)

                label_pred = decoder(output)

                label_pred[0] = round(label_pred[0], 10)

                test_precision = precision_score(np.array(label_pred), np.array(labels.tolist()), average='micro')
                test_precision_average.append(test_precision)

    experiment.log_metric('test_loss', test_loss, step=iter_meter.get())
    experiment.log_metric('test_precision', np.mean(test_precision_average), step=iter_meter.get())

    print('Test set: Average loss: {:.4f}\tTest precision: {:.2f}%\n'.format(
        test_loss, 100*np.mean(test_precision_average)))

    return np.mean(test_precision_average)


if __name__ == "__main__":

    # Create an experiment with your api key
    experiment = Experiment(
        api_key="NXte5ivrmxMfwBeYjvahf97PC",
        project_name="food-number",
        workspace="hai321",
    )

    experiment.add_tags(["confirm_data", "deep_speech_model"])
    experiment.set_name("Test confirm data with deepspeech model")

    experiment.log_parameters(FoodNumberModel.hparams)

    # Config gpu/cpu
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(7)
    device = torch.device("cuda" if use_cuda else "cpu")

    model = FoodNumberModel(FoodNumberModel.hparams['n_cnn_layers'], FoodNumberModel.hparams['n_rnn_layers'], FoodNumberModel.hparams['rnn_dim'], FoodNumberModel.hparams['n_class'], FoodNumberModel.hparams['n_feats'], \
        FoodNumberModel.hparams['stride'], FoodNumberModel.hparams['dropout']).to(device)

    try:
        checkpoint = torch.load(SAVED_MODEL_PATH)
        model.load_state_dict(checkpoint)
    except:
        pass

    # model summaries
    print(model)
    print('Num Model Parameters', sum(
        [param.nelement() for param in model.parameters()]))

    optimizer = optim.AdamW(
        model.parameters(), FoodNumberModel.hparams["learning_rate"])
    criterion = nn.CrossEntropyLoss().to(device)

    iter_meter = IterMeter()
    try:
        for epoch in range(1, FoodNumberModel.hparams["epochs"] + 1):
            precision_average = []

            for data_path in DATA_PATH:
                filename = data_path.split("/")[-1]
                # Load all data
                mel_spectrogram, labels = load_data(data_path)

                # Split into train and test
                mel_spectrogram_train, mel_spectrogram_test, labels_train, labels_test = train_test_split(mel_spectrogram, labels, test_size=FoodNumberModel.hparams['test_size'], shuffle=False)
                

                # Create train dataset and Dataloader
                train_dataset = Dataset(
                    mel_spectrogram_train, labels_train)

                train_loader = data.DataLoader(dataset=train_dataset,
                                            batch_size=FoodNumberModel.hparams["batch_size"],
                                            shuffle=True if epoch>10 else False)

                # Create test dataset and Dataloader
                test_dataset = Dataset(mel_spectrogram_test, labels_test)

                test_loader = data.DataLoader(dataset=test_dataset,
                                            batch_size=FoodNumberModel.hparams["batch_size"],
                                            shuffle=True if epoch>10 else False)

                scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=FoodNumberModel.hparams["learning_rate"],
                                                        steps_per_epoch=int(
                                                            len(train_loader)),
                                                        epochs=FoodNumberModel.hparams["epochs"],
                                                        anneal_strategy='linear')

                train(model, device, train_loader, criterion, optimizer,
                    scheduler, epoch, iter_meter, experiment)

                precision = test(model, device, test_loader, criterion,
                    iter_meter, experiment, filename)

                precision_average.append(precision)

            if np.mean(precision_average) > 0.9:
                raise GetOutOfLoop

    except GetOutOfLoop:
        pass

    # Save model
    torch.save(model.state_dict(), SAVED_MODEL_PATH)
