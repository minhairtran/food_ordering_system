import sys
sys.path.append("../../food_ordering_system")
sys.path.append(
    "/home/minhair/Desktop/food_ordering_system/test_pytorch_venv/lib/python3.8/site-packages/")
from comet_ml import Experiment

from sklearn.model_selection import train_test_split
import numpy as np
import torch.optim as optim
import torch.utils.data as data
import torch.nn as nn
import torch
from sklearn.metrics import precision_score
from model import Food_model


DATA_PATH = ["../data/food_data/data_set_0.pt", "../data/food_data/data_set_1.pt", "../data/food_data/data_set_2.pt",
             "../data/food_data/data_set_3.pt", "../data/food_data/data_set_4.pt", "../data/food_data/data_set_5.pt",
             "../data/food_data/data_set_6.pt", "../data/food_data/data_set_7.pt", "../data/food_data/data_set_8.pt",
             "../data/food_data/data_set_9.pt", "../data/food_data/data_set_10.pt", "../data/food_data/data_set_11.pt",
             "../data/food_data/data_set_12.pt", "../data/food_data/data_set_13.pt"]
SAVED_MODEL_PATH = "model_food_number.h5"
class TrainingSuccess(Exception):
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

def train(model, device, train_loader, criterion, optimizer, epoch, iter_meter, experiment):
    model.train()
    data_len = len(train_loader.dataset)

    with experiment.train():
        for batch_idx, _data in enumerate(train_loader):
            spectrograms, labels = _data
            spectrograms, labels = spectrograms.to(device), labels.to(
                device)  # spectro (batch, cnn_feature, n_class, time)

            optimizer.zero_grad()

            output = model(spectrograms)  # (batch, time, n_class)

            loss = criterion(output, labels)
            loss.backward()

            experiment.log_metric('loss', loss.item(), step=iter_meter.get())

            optimizer.step()
            iter_meter.step()

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(spectrograms), data_len,
                100. * (batch_idx + 1) / len(train_loader), loss.item()))

def test(model, device, test_loader, criterion, iter_meter, experiment, filename):
    print('\nEvaluating ' + str(filename) + "...")
    model.eval()
    test_precision_average = []

    epoch_loss = 0

    with experiment.test():
        with torch.no_grad():
            for i, _data in enumerate(test_loader):
                spectrograms, labels = _data
                spectrograms, labels = spectrograms.to(device), labels.to(
                    device)  # spectro (batch, cnn_feature, n_class, time)
                output = model(spectrograms)  # (batch, time, n_class)

                preds = torch.argmax(output, 1).tolist()

                loss = criterion(output, labels)

                epoch_loss += loss.item() * spectrograms.size(0)

                test_precision = precision_score(np.array(labels.tolist()), np.array(preds), average='micro')
                test_precision_average.append(test_precision)
                

    experiment.log_metric('test_loss', epoch_loss, step=iter_meter.get())

    print('Test set: Average loss: {:.4f}\tTest precision: {:.2f}%\n'.format(
        epoch_loss, 100*np.mean(test_precision_average)))

    return np.mean(test_precision_average), epoch_loss


if __name__ == "__main__":

    # Create an experiment with your api key
    experiment = Experiment(
        api_key="NXte5ivrmxMfwBeYjvahf97PC",
        project_name="food",
        workspace="hai321",
    )

    experiment.add_tags(["food_data", "attiontion-based"])
    experiment.set_name("(Freq_mask; Time_mask) = (13;5)")

    experiment.log_parameters(Food_model.hparams)

    # Config gpu/cpu
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(7)
    device = torch.device("cuda" if use_cuda else "cpu")

    model = Food_model(Food_model.hparams['n_mels'], Food_model.hparams['cnn_channels'], Food_model.hparams['cnn_kernel_size'], \
        Food_model.hparams['stride'], Food_model.hparams['gru_hidden_size'], Food_model.hparams['attention_hidden_size'], Food_model.hparams['n_classes']).to(device)

    try:
        checkpoint = torch.load(SAVED_MODEL_PATH)
        model.load_state_dict(checkpoint)
    except:
        pass

    # model summaries
    print(model)
    print('Num Model Parameters', sum(
        [param.nelement() for param in model.parameters()]))

    optimizer = optim.Adam(model.parameters(), Food_model.hparams["learning_rate"])

    criterion = nn.NLLLoss().to(device)

    iter_meter = IterMeter()
    precision = 0
    max_precision = 0
    model_saved_message = ''
    
    try:
        for epoch in range(1, Food_model.hparams["epochs"] + 1):
            epoch_precisions = []
            epoch_loss = []

            for data_path in DATA_PATH:
                filename = data_path.split("/")[-1]
                mel_spectrogram, labels = load_data(data_path)

                # Split into train and test
                mel_spectrogram_train, mel_spectrogram_test, labels_train, labels_test = train_test_split(mel_spectrogram, labels, test_size=Food_model.hparams['test_size'], shuffle=False)

                # Create train dataset and Dataloader
                train_dataset = Dataset(mel_spectrogram_train, labels_train)

                train_loader = data.DataLoader(dataset=train_dataset,
                                            batch_size=Food_model.hparams["batch_size"],
                                            shuffle=True)

                # Create test dataset and Dataloader
                test_dataset = Dataset(mel_spectrogram_test, labels_test)

                test_loader = data.DataLoader(dataset=test_dataset,
                                            batch_size=Food_model.hparams["batch_size"],
                                            shuffle=True)
                
                train(model, device, train_loader, criterion, optimizer, epoch, iter_meter, experiment)

                precision, loss = test(model, device, test_loader, criterion,
                    iter_meter, experiment, filename)

                epoch_precisions.append(precision)
                epoch_loss.append(loss)

            print('Test set: Test precision: {:.2f}%\n'.format(100*np.mean(epoch_precisions)))

            with experiment.test():
                experiment.log_metric('test_precision', np.mean(epoch_precisions), step=iter_meter.get())
            # Save model
            if np.mean(epoch_precisions) > max_precision:
                max_precision = np.mean(epoch_precisions)
                torch.save(model.state_dict(), SAVED_MODEL_PATH)
                model_saved_message = "Model saved at test_precision: " + str(max_precision)

            if np.mean(epoch_precisions) > 0.99:
                raise TrainingSuccess

    except TrainingSuccess:
        pass
    finally:
        print(model_saved_message)
