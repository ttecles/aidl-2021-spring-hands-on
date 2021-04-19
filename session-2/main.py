import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import ray
from ray import tune
from dataset import MyDataset
from model import MyModel
from utils import accuracy, save_model


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

if not torch.cuda.is_available():
    raise Exception("no CUDA available")

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

    print(f"Epoch {epoch+1} loss: {loss.item():.2f}")


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
    print(f"Epoch {epoch+1} accuracy: {correct / total*100:.1f}%")


def train_model(config):
    train_loader, eval_loader, test_dataset = data_setup(config)

    my_model = MyModel(config['conv_kern'], config['mlp_neurons'], config['dropout']).to(device)
    for epoch in range(config["epochs"]):
        train_single_epoch(epoch, my_model, train_loader, config['learning_rate'])
        eval_single_epoch(epoch, my_model, eval_loader)

    return my_model, test_dataset


def test_model(model, test_dataset):
    model.eval()
    for input, label in test_dataset:
        inputs = input.unsqueeze(0).to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        val = int(predicted[0])
        print(f"predicted={val} real={label}")


def data_setup(config):
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

    return train_loader, eval_loader, test_dataset

if __name__ == "__main__":

    use_ray_tune = True


    if use_ray_tune:

        ray.init(configure_logging=False)
        analysis = tune.run(
            train_model,
            metric="val_loss",
            mode="min",
            num_samples=5,
            config={
                "train_dataset_factor": 0.7,
                "test_count": 20,
                "loader_workers": 2,
                "epochs": 10,
                "batch_size": tune.randint(32, 128),
                "learning_rate": tune.loguniform(1e-4, 1e-1),
                "conv_kern": tune.randint(16, 128),
                "mlp_neurons": tune.randint(100, 150), 
                "dropout": tune.grid_search([0.1, 0.2, 0.3, 0.4, 0.5]),
            })

        print(analysis)
    else:
        config = {
            "train_dataset_factor": 0.7,
            "test_count": 20,
            "loader_workers": 2,
            "batch_size": 32,
            "epochs": 10,
            "learning_rate": 0.001,
            "conv_kern": 64,
            "mlp_neurons": 120,
            "dropout": 0.5,
        }
        model, test_dataset = train_model(config)
        test_model(model, test_dataset)


