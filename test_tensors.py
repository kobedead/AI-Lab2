import torch.nn.functional

import run_tensors as run_tensors
from run_tensors import mse, plot_rgb_tensor


class Options:
    def __init__(self):
        # runtime related
        self.random_seed = 1
        self.device = "cpu"

        # model related
        self.save_path = "./models/"
        self.load_path = "./models/"
        self.model_name = "lin_reg.pth"

def test_mse():
    test_input = torch.randn([3, 50, 100])
    test_target = torch.randn([3, 50, 100])
    error = mse(test_input, test_target)
    if error == "NOT IMPLEMENTED":
        print("MSE not yet implemented... (0/1)")
    elif error.shape != torch.Size([]):
        print("MSE not correctly implemented... (0/1)")
        print("The loss shape is incorrect.")
        print(f"Yours: {error.shape}")
        print(f"Correct: {torch.Size([])}")
    else:
        error2 = torch.nn.functional.mse_loss(test_input, test_target)
        if error.item() == error2.item():
            print("MSE implemented correctly! (1/1)")
        else:
            print("MSE not correctly implemented... (0/1)")
            print("Hint: did you forget to take the mean?")


def test_image_matrix():
    created_image = run_tensors.create_image()
    if created_image == "NOT IMPLEMENTED":
        print("create_image not yet implemented... (0/1)")
    else:
        good_image = torch.load("images/create_image_tensor.pt")
        plot_rgb_tensor(good_image, "Target")
        plot_rgb_tensor(created_image, "Yours")
        if torch.equal(good_image, created_image):
            print("create_image implemented correctly! (1/1)")
        else:
            print("create_image not implemented correctly... (0/1)")
            print(f"Your image shape: {created_image.shape}\nOur image shape: {good_image.shape}")
            print(created_image)
            print(good_image)
            print(f"If the shapes match, perhaps there is an error in the pixel values...")


def test_tensor_forward():
    input_tensor = torch.FloatTensor([0.2, 0.1, 0.5, 0.9], device=options.device)
    weights = torch.FloatTensor([0.4, -0.1, 0.4, -0.5], device=options.device)
    output = run_tensors.lin_layer_forward(weights, input_tensor)
    if output == "NOT IMPLEMENTED":
        print("tensor_forward not yet implemented... (0/1)")
    elif output.shape != torch.Size([]):
        print("Something is wrong with the output shape... (0/1)")
        print(f"Yours: {output.shape}")
        print(f"Target: {torch.Size([])}")
    else:
        if output.item() - (-0.18) < 0.000001:
            print("tensor_forward implemented correctly! (1/1)")
        else:
            print("tensor_forward not implemented correctly... (0/1)")




if __name__ == "__main__":




    options = Options()
    print("\nQUESTION 1:\n")
    test_mse()
    print("\nQUESTION 2:\n")
    test_image_matrix()
    print("\nQUESTION 3:\n")
    test_tensor_forward()
