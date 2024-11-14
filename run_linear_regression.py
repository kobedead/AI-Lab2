import os
import torch
import matplotlib.pyplot as plt

from datasets import HousesDataset

from models import LinearRegressionModel

from learners import RegressionLearner

from utilities import plot_house_sizes_and_prices, plot_house_results, plot_training_process


## PARAMETERS

random_seed = 1

batch_size = 32

epochs = 10

learning_rate = 0.1

## INITIALISATION

# manually set seeds for random number generators --> ensures reproducibility of results

torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)

## HARDWARE ACCELERATION

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

print('device:', device)
print()

# load the houses dataset

train_dataset = HousesDataset(train=True)
test_dataset = HousesDataset(train=False)

print("dataset dimensions:")
print()
print("train data shape:", tuple(train_dataset.data.shape))
print("test data shape:", tuple(test_dataset.data.shape))
print()

plot_house_sizes_and_prices(train_dataset[:][0], train_dataset[:][1], title="Train dataset")
plot_house_sizes_and_prices(test_dataset[:][0], test_dataset[:][1], title="Test dataset")

# create a dataloaders that sample batches from the train dataset
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size)

## MODEL DEFINITION

"""START TODO: fill in the missing parts"""

# create an instance of the linear regression model

model = LinearRegressionModel()


# define an opimizer to fit the model (see https://pytorch.org/docs/stable/optim.html)

optimizer = torch.optim.Adam(model.parameters() ,  lr=learning_rate)

"""END TODO"""

print("linear model architecture:")
print()
print(model)
print()

learner = RegressionLearner(model, optimizer)

## TRAINING

# train the model
train_losses = learner.train(train_dataloader, epochs, device)

# plot the training loss versus the validation loss
plot_training_process(train_losses, [])

## TESTING

# load the best version of the model parameters
learner.load("leader")

# evaluate the model on the train dataset
train_loss = learner.test(train_dataset, device)
train_results = learner.predict(train_dataset, device).cpu()
 
print(f"performance on the train dataset:\troot mean squared error = {train_loss:.2f} €")
print()

# evaluate the model on the test dataset
test_loss = learner.test(test_dataset, device)
test_results = learner.predict(test_dataset, device).cpu()

print(f"performance on the test dataset:\troot mean squared error = {test_loss:.2f} €")
print()

# evaluate the model on the "unknown" linear function f(x)
linear_sizes = torch.linspace(train_dataset.min_house_size, train_dataset.max_house_size, 1000)
linear_prices = 5000 * linear_sizes + 100000
linear_dataset = (linear_sizes.unsqueeze(1), linear_prices.unsqueeze(1))
linear_results = learner.predict(linear_dataset, device).cpu()

plot_house_results(train_dataset.data, train_dataset.targets, train_results, linear_sizes, linear_prices, linear_results, title="Performance on train dataset")

plot_house_results(test_dataset.data, test_dataset.targets, test_results, linear_sizes, linear_prices, linear_results, title="Performance on test dataset")

print(f"unknown function: f(x) = 5000 * x + 100 000 + {test_dataset.noise_house_price} * N(0, 1).")
print(f"learned function: g(x) = {model.linear_layer.weight.data[0].item()} * x + {model.linear_layer.bias.data[0].item()}")
print()