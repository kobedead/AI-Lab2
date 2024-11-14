import os
import math
import torch

from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader

from run_tensors import mse

class Learner():

    def __init__(self, model: Module, criterion: Module, optimizer: Optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    def save(self, label):
        if not os.path.exists(os.path.join("checkpoints")):
            os.makedirs(os.path.join("checkpoints"))
        torch.save(self.model.state_dict(), os.path.join("checkpoints", self.model.__class__.__name__ + "_" + label))

    def load(self, label):
        self.model.load_state_dict(torch.load(os.path.join("checkpoints", self.model.__class__.__name__ + "_" + label), weights_only=True))
    

class RegressionLearner(Learner):

    def __init__(self, model: Module, optimizer: Optimizer):
        super().__init__(model, mse, optimizer)

    def train(self, dataloader: DataLoader, epochs: int, device: torch.device):

        self.model.to(device)

        train_losses = []

        for epoch in range(epochs):

            self.model.train()  # set model in training mode

            train_loss = 0

            for data, targets in dataloader:
                data = data.to(device)
                targets = targets.to(device)

                """START TODO: fill in the missing parts"""

                # forward the size data through the model

                next = self.model.forward(data)

                # calculate the loss using the self-implemented mean squared error function
                loss = mse(next , targets)

                # As mentioned before, the grads always needs to be zeroed before backprop (use your optimizer to do this)
                self.optimizer.zero_grad()

                # propagate the loss backward
                loss.backward()

                # use your optimizer to perform an update step

                self.optimizer.step()

                """END TODO"""

                train_loss += loss.item() * len(data)

            train_loss = math.sqrt(train_loss / len(dataloader.dataset))  # calculates the root of the mean of the square errors

            train_losses.append(train_loss)

          #  print(f"train epoch [{epoch + 1}/{epochs}]:\ttrain loss = {train_loss:.2f} €")

            if train_loss <= min(train_losses):
                self.save("leader")

        print()
        
        self.save("final")

        return train_losses

    @torch.no_grad()
    def test(self, dataset: Dataset, device: torch.device):

        self.model.to(device)

        self.model.eval()  # set model in evaluation mode

        test_loss = 0

        """START TODO: fill in the missing parts"""

        # forward the data through the model

        for data,target in dataset :
            output = self.model.forward(data)

        # calculate the loss using the self-implemented mean squared error function

            test_loss =+ mse(output , target)

        # calculate the root mean squared error for the dataset
        print(len(dataset))
        test_loss = math.sqrt(test_loss/len(dataset))


        """END TODO"""

        return test_loss

    @torch.no_grad()
    def predict(self, dataset, device):
        self.model.to(device)
        self.model.eval()
        data = dataset[:][0]
        data = data.to(device)
        results = self.model(data)
        return results


class ClassificationLearner(Learner):

    def __init__(self, model: Module, criterion: Module, optimizer: Optimizer):
        super().__init__(model, criterion, optimizer)

    def train(self, train_dataloader: DataLoader, valid_dataloader: DataLoader, epochs: int, device: torch.device):

        self.model.to(device)

        train_losses = []
        valid_losses = []
        valid_accuracies = []

        for epoch in range(epochs):
            
            self.model.train()  # set model in training mode

            train_loss = 0

            for data, targets in train_dataloader:
                data = data.to(device)
                targets = targets.to(device)

                """START TODO: fill in the missing parts"""

                # forward the data through the model



                output = self.model.forward(data)


                # calculate the loss
                loss = self.criterion(output, targets)

                # set all gradients to zero
                self.optimizer.zero_grad()

                # propagate the loss backwards

                loss.backward()

                # use your optimizer to perform an update step

                self.optimizer.step()

                """END TODO"""

                train_loss += loss.item() * len(data)

            train_loss = train_loss / len(train_dataloader.dataset)

            self.model.eval()  # set model in evaluation mode

            # evaluate the model on the validation dataset
            valid_loss, valid_accuracy = self._evaluate(valid_dataloader, device)    

            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            valid_accuracies.append(valid_accuracy)

            print(f"train epoch [{epoch + 1}/{epochs}]:\ttrain loss = {train_loss:.4f}\tvalid loss = {valid_loss:.4f}\tvalid accuracy = {100 * valid_accuracy:.2f}%")   

            if valid_loss <= min(valid_losses):
                self.save("leader")

        print()
        
        self.save("final")

        return (train_losses, valid_losses, valid_accuracies)

    def test(self, test_dataloader: DataLoader, device: torch.device):

        self.model.to(device)

        self.model.eval()  # set model in evaluation mode

        # evaluate the model on the test dataset
        test_loss, test_accuracy = self._evaluate(test_dataloader, device)

        return (test_loss, test_accuracy)

    @torch.no_grad()
    def _evaluate(self, dataloader, device):
        eval_loss = 0
        eval_accuracy = 0

        """START TODO: fill in the missing parts"""

        # process the data in batches
        for data , targets in dataloader :
            data = data.to(device)
            targets = targets.to(device)
            # forward the data through the model
            out = self.model.forward(data)

            # calculate the loss
            loss = self.criterion(out , targets)

            # predict the labels (index of 1 -> (0-9))
            predLabels = torch.argmax(out , dim=1)


            # calculate the evaluation loss and accuracy for a batch
            for i in range(len(predLabels)) :
                if predLabels[i]==targets[i]:
                    eval_accuracy = eval_accuracy + 1

            eval_loss += loss.item()*len(data)

        # calculate the evaluation loss and accuracy for the dataset
        eval_loss = eval_loss/len(dataloader.dataset)
        eval_accuracy = eval_accuracy/len(dataloader.dataset)



        """END TODO"""

        return (eval_loss, eval_accuracy)


