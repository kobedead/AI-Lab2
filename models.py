import torch




class LinearRegressionModel(torch.nn.Module):
    def __init__(self) -> None :
        super().__init__()
        """START TODO: define a linear layer"""

        self.linear_layer = torch.nn.Linear(1,1)
        #self.linear_layer2 = torch.nn.Linear(10,1)


        """END TODO"""

    def forward(self, x: torch.Tensor):
        """START TODO: forward the tensor x through the linear layer and return the output"""

        x = self.linear_layer(x)
        #x = self.linear_layer2(x)

        """END TODO"""
        return x


class NeuralNetworkClassificationModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        """ START TODO: fill in all three layers. Remember that each layer should contain 2 parts, a linear transformation and a nonlinear activation function."""
        self.layer1 = None
        self.layer2 = None
        self.layer3 = None
        """END TODO"""

    def forward(self, x: torch.Tensor):
        """START TODO: forward tensor x through all layers."""
        
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
