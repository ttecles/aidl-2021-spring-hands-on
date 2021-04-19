import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms

from dataset import MyDataset
from model import MyModel
from utils import accuracy, save_model


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def train_single_epoch(epoch, model, train_loader, lr):
    criterion = F.nll_loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    for i, (inputs, labels) in enumerate(train_loader):
        # get the inputs to gpu; data is a list of [inputs, labels]
        inputs, labels = inputs.to(device), labels.to(device)

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print(f"Epoch {epoch+1} [{len(train_loader)}] loss: {loss.item():.2f}")


def eval_single_epoch(epoch, model, eval_loader):
    correct = 0
    total = 0
    model.eval()
    for inputs, labels in eval_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print(f"Epoch {epoch+1} accuracy of the network: {correct / total*100:.1f}%")


def train_model(config, train_loader, eval_loader):
    my_model = MyModel(config['conv_kern'], config['mlp_neurons']).to(device)
    for epoch in range(config["epochs"]):
        train_single_epoch(epoch, my_model, train_loader, config['learning_rate'])
        eval_single_epoch(epoch, my_model, eval_loader)

    return my_model


def test_model(model, test_dataset):
    model.eval()
    for input, label in test_dataset:
        inputs = input.unsqueeze(0).to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        val = int(predicted[0])
        print(f"predicted={val} real={label}")

if __name__ == "__main__":

    config = {
        "train_dataset_factor": 0.7,
        "test_count": 20,
        "loader_workers": 2,
        "epochs": 10,
        "batch_size": 32,
        "learning_rate": 0.001,
        "dropout": 1,
        "conv_kern": 64,
        "mlp_neurons": 120,
    }

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, ), (0.5, ))])

    my_dataset = MyDataset("dataset/data", "dataset/chinese_mnist.csv", transform)
    test_count =  int(config["test_count"])
    total_count = len(my_dataset) - test_count
    train_count = int(total_count*config["train_dataset_factor"]) 
    eval_count = total_count - train_count

    train_dataset, eval_dataset, test_dataset = torch.utils.data.random_split(my_dataset,
        (train_count, eval_count, test_count))

    def create_loader(dataset):
        return torch.utils.data.DataLoader(dataset, config['batch_size'],
            shuffle=True, num_workers=config["loader_workers"])

    train_loader = create_loader(train_dataset)
    eval_loader = create_loader(eval_dataset)

    model = train_model(config, train_loader, eval_loader)

    test_model(model, test_dataset)


