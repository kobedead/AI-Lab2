import torch




class LinearRegressionModel(torch.nn.Module):
    def __init__(self) -> None :
        super().__init__()
        """START TODO: define a linear layer"""

        self.linear_layer = torch.nn.Linear(1,1)


        """END TODO"""

    def forward(self, x: torch.Tensor):
        """START TODO: forward the tensor x through the linear layer and return the output"""

        x = self.linear_layer(x)

        """END TODO"""
        return x


class NeuralNetworkClassificationModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        """ START TODO: fill in all three layers. Remember that each layer should contain 2 parts, a linear transformation and a nonlinear activation function."""
                            #image size=28*28
        self.layer1 = torch.nn.Linear(28*28  , 500)
        self.layer2 = torch.nn.Linear(500  , 200)
        self.layer3 = torch.nn.Linear(200  , 10)

        self.seq = torch.nn.Sequential(self.layer1 ,
                                       torch.nn.ReLU(),
                                       self.layer2 ,
                                       torch.nn.ReLU(),
                                       self.layer3 ,
                                       torch.nn.LogSoftmax(dim=1)
                                       )


        """END TODO"""

    def forward(self, x: torch.Tensor):
        """START TODO: forward tensor x through all layers."""

        x = self.seq(x.view(len(x), -1))


        """
        x = self.layer1(x.view(len(x), -1))
        x = torch.nn.ReLU(x)
        x = self.layer2(x)
        x = torch.nn.ReLU(x)
        x = self.layer3(x)
        x = torch.nn.Softmax(x)
        """
        """END TODO"""
        return x


class NeuralNetworkClassificationModelWithVariableLayers(torch.nn.Module):
    def __init__(self, in_size, out_size, hidden_sizes = []):
        super().__init__()

        self.layers = torch.nn.Sequential()

        for i, size in enumerate(hidden_sizes):
            self.layers.add_module(f"lin_layer_{i + 1}", torch.nn.Linear(in_size, size))
            self.layers.add_module(f"relu_layer_{i + 1}", torch.nn.ReLU())
            in_size = size

        self.layers.add_module(f"lin_layer_{len(hidden_sizes) + 1}", torch.nn.Linear(in_size, out_size))
        self.layers.add_module(f"softmax_layer", torch.nn.Softmax(dim=1))

    def forward(self, x: torch.Tensor):
        x = x.flatten(1)
        x = self.layers(x)
        return x
