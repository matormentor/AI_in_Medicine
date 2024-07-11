import torch
import sys
import flwr as fl

from centralized import load_model, load_data, load_datasets, train, test
from collections import OrderedDict

# Define parameters
EPOCH_NUM = 20
NUM_CLIENTS = 6


def set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


class Client(fl.client.NumPyClient):
    def __init__(self, cid, net, train_loader, val_loader):
        self.cid = cid
        self.net = net
        self.train_loader = train_loader
        self.val_loader = val_loader
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in model.state_dict().items()]

    def fit(self, parameters, config):
        set_parameters(model, parameters)
        train(self.train_loader, self.val_loader, model, epochs=EPOCH_NUM)
        return self.get_parameters(config={}), len(self.train_loader), {}

    def evaluate(self, parameters, config):
        set_parameters(model, parameters)
        loss, accuracy = test(self.val_loader, model)
        return float(loss), len(self.val_loader), {"accuracy": float(accuracy)}


if __name__ == "__main__":
    n = len(sys.argv)
    if n == 2:
        cid = int(sys.argv[1])
        print("Client defined:", cid)
    else:
        sys.exit()

    model = load_model()
    train_loaders, val_loaders, test_loader = load_datasets(NUM_CLIENTS)

    train_loader = train_loaders[int(cid)]
    val_loader = val_loaders[int(cid)]

    client = Client(cid, model, train_loader, val_loader)
    fl.client.start_numpy_client(server_address="localhost:8080", client=client)
