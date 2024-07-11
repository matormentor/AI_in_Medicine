import torch
import torch.nn as nn
import torch.optim as optim
import medmnist
import torchvision.transforms as transforms

from torch.utils.data import random_split
from torch.utils.data import DataLoader
from medmnist import INFO
from tqdm import tqdm
from sklearn.metrics import accuracy_score

# Defining global variables
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
LR = 2e-4
BATCH_SIZE = 32
EPOCH_NUM = 20


# Defining the network
class CNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(CNN, self).__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes

        self.activation = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, 12, 3),  # 26
            self.activation,
            nn.BatchNorm2d(12),
            nn.Conv2d(12, 24, 3),  # 24
            self.activation,
            nn.BatchNorm2d(24),
            self.pool,  # 12
            nn.Conv2d(24, 48, 3),  # 10
            self.activation,
            nn.BatchNorm2d(48),
            nn.Conv2d(48, 96, 3),  # 8
            self.activation,
            nn.BatchNorm2d(96),
            self.pool,  # 4
            nn.Conv2d(96, 512, 3),  # 2
            self.activation,
            nn.BatchNorm2d(512),
            self.pool,  # 1
        )
        self.fc_block = nn.Sequential(
            nn.Linear(512 * 1 * 1, 1024),  # fc1
            self.activation,
            nn.Linear(1024, 1024),  # fc2
            self.activation,
            nn.Linear(1024, num_classes),  # output layer (not counted as individual fc layer, according to convention)
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = torch.flatten(x, 1)
        x = self.fc_block(x)

        return x


# Defining the training function
def train(train_loader, val_loader, model, epochs):
    loss_values = []
    val_loss_values = []

    model.train()

    optimizer = torch.optim.Adam(model.parameters())
    # optimizer = torch.optim.SGD(params=model.parameters(), lr=LR)
    criterion = torch.nn.BCELoss()

    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=epochs * len(train_loader) / 10, gamma=.75, verbose=False)

    for epoch in range(epochs):
        # training
        training_loop = tqdm(enumerate(train_loader), total=len(train_loader), ncols=150,
                             desc='Training Epoch     [%02i/%02i]' % (epoch + 1, epochs))
        train_loss = 0

        for iteration, batch in training_loop:
            # outputs
            optimizer.zero_grad()
            x, y = batch
            y = y.float()
            x, y = x.to(DEVICE, dtype=torch.float32), y.to(DEVICE, dtype=torch.float32)
            y_pred = model(x)
            # loss
            loss = criterion(y_pred, y)
            train_loss += loss
            # backprop
            loss.backward()
            optimizer.step()
            # update loop
            training_loop.set_postfix(train_loss="{:.8f}".format(train_loss / (iteration + 1)))
            scheduler.step()

        loss_values.append(train_loss.item() / len(training_loop))

        # validation
        model.eval()
        val_loop = tqdm(enumerate(val_loader), total=len(val_loader), ncols=150,
                        desc='Validation Epoch   [%02i/%02i]' % (epoch + 1, epochs))
        val_loss = 0

        with torch.no_grad():
            for iteration, batch in val_loop:
                # outputs
                x, y = batch;
                y = y.float()
                x, y = x.to(DEVICE, dtype=torch.float32), y.to(DEVICE, dtype=torch.float32)
                y_pred = model(x)
                # loss
                loss = criterion(y_pred, y)
                val_loss += loss
                # update loop
                val_loop.set_postfix(val_loss="{:.8f}".format(val_loss / (iteration + 1)))

            val_loss_values.append(val_loss.item() / len(val_loop))

    return loss_values, val_loss_values, model


# Define the test function
def test(test_loader, model):
    criterion = torch.nn.BCELoss()
    y_list = []
    y_pred_list = []

    model.eval()

    acc, loss = 0., 0.
    acc_list = list()
    loss_list = list()
    verbose = True

    with torch.no_grad():
        for iteration, batch in enumerate(test_loader):
            # outputs
            x, y = batch;
            y = y.float()
            y_list.append(y)
            x, y = x.to(DEVICE, dtype=torch.float32), y.to(DEVICE, dtype=torch.float32)
            y_pred = model(x)
            y_pred_list.append(y_pred)
            # loss
            loss_batch = criterion(y_pred, y)
            loss_list.append(loss_batch.item())
            y_pred_binary = torch.where(y_pred.cpu() > 0.5, 1, 0)
            acc_batch = accuracy_score(y.cpu(), y_pred_binary)
            acc += acc_batch
            loss += loss_batch
            acc_list.append(acc_batch)

        acc /= len(test_loader)
        loss /= len(test_loader)

    # check metrics for every batch
    if verbose:
        print('\n#----------------------#')
        print('   Batch  Loss    Acc')
        for index in range(len(test_loader)):
            print('      #{:d}  {:.3f}  {:.3f}'.format(
                index + 1, loss_list[index], acc_list[index]
            ))
        print('#----------------------#\n')

    return loss, acc


def data_parameter_setter():
    data_flag_2d = 'breastmnist'  # here the name of dataset should be passed.
    download = True

    info = INFO[data_flag_2d]
    print(info)
    task = info['task']
    n_channels = info['n_channels']
    n_classes = 1

    return n_channels, n_classes, download, info


def load_data():
    # preprocessing such as conversion to tensor and normalization
    _, _, download, info = data_parameter_setter()
    trafo = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])  # output (-1, 1)

    # get the data class
    DataClass = getattr(medmnist, info['python_class'])
    train_dataset = DataClass(split='train', transform=trafo, download=download)
    val_dataset = DataClass(split='val', transform=trafo, download=download)
    test_dataset = DataClass(split='test', transform=trafo, download=download)

    # encapsulate datasets into Dataloader form
    train_loader = DataLoader(train_dataset, BATCH_SIZE)
    val_loader = DataLoader(val_dataset, BATCH_SIZE)
    test_loader = DataLoader(test_dataset, BATCH_SIZE)
    num_examples = {"trainset": len(train_dataset), "testset": len(test_dataset)}

    return train_loader, test_loader, val_loader, num_examples

def load_datasets(num_clients):
    # preprocessing such as conversion to tensor and normalization
    _, _, download, info = data_parameter_setter()
    trafo = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])  # output (-1, 1)

    # get the data class
    DataClass = getattr(medmnist, info['python_class'])
    train_dataset = DataClass(split='train', transform=trafo, download=download)
    val_dataset = DataClass(split='val', transform=trafo, download=download)
    test_dataset = DataClass(split='test', transform=trafo, download=download)

    # Split training set into `num_clients` partitions to simulate different local datasets
    partition_size_train = len(train_dataset) // num_clients
    lengths_train = [partition_size_train] * num_clients
    partition_size_val = len(val_dataset) // num_clients
    lengths_val = [partition_size_val] * num_clients
    print(len(train_dataset), partition_size_train, lengths_train, sep="\n")
    datasets_train = random_split(train_dataset, lengths_train, torch.Generator().manual_seed(42))
    datasets_val = random_split(val_dataset, lengths_val, torch.Generator().manual_seed(42))

    # Split each partition into train/val and create DataLoader
    train_loaders = []
    val_loaders = []
    for c in range(num_clients):
        train_loaders.append(DataLoader(datasets_train[c], batch_size=BATCH_SIZE, shuffle=True))
        val_loaders.append(DataLoader(datasets_val[c], batch_size=BATCH_SIZE))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    return train_loaders, val_loaders, test_loader


def load_model():
    n_channels, n_classes, _, _ = data_parameter_setter()
    model = CNN(in_channels=n_channels, num_classes=n_classes)
    return model.to(DEVICE)


if __name__ == "__main__":
    # Set parameters
    train_loader, test_loader, val_loader, num_examples = load_data()
    model = load_model()

    # Train model
    loss_values, val_loss_values, model = train(train_loader, val_loader, model, epochs=EPOCH_NUM)

    # Test model
    loss, acc = test(test_loader, model)

    print('Mean loss: {:.3f}, mean acc: {:.3f}'.format(loss, acc))
