import os
import torch
import numpy as np
from typing import Dict, Union, Optional
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

class TensorBoardLogger:
    """
    A unified logging utility for PyTorch models using TensorBoard.

    This class wraps the torch.utils.tensorboard.SummaryWriter to provide 
    a cleaner and more structured interface for logging scalars, images, 
    histograms, and other data during training.
    """

    def __init__(self, log_dir: str, experiment_name: str = "default_experiment"):
        """
        Initializes the TensorBoardLogger.

        Args:
            log_dir (str): The base directory where logs will be saved.
                           A subdirectory with the experiment_name will be created here.
            experiment_name (str): A unique name for the current training run.
        """
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.log_path = os.path.join(log_dir, experiment_name)
        self.time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        # Create the SummaryWriter instance
        self.writer = SummaryWriter(log_dir=self.log_path)
        print(f"TensorBoard logs are being saved to: {self.log_path}")

    def log_scalar(self, tag: str, value: Union[float, int, torch.Tensor], step: int):
        """
        Logs a single scalar value.

        Args:
            tag (str): Data identifier (e.g., 'Loss/train', 'Accuracy/validation').
            value (Union[float, int, torch.Tensor]): The scalar value to log.
            step (int): Global step or epoch number for the x-axis.
        """
        if isinstance(value, torch.Tensor):
            value = value.item() # Get the Python number from the tensor
        self.writer.add_scalar(tag, value, step)

    def log_scalars(self, tag_value_dict: Dict[str, Union[float, int, torch.Tensor]], step: int):
        """
        Logs multiple scalar values at once under a common parent tag.

        Args:
            tag_value_dict (Dict[str, Union[float, int, torch.Tensor]]): 
                A dictionary where keys are sub-tags (e.g., 'learning_rate') and 
                values are the scalar data.
            step (int): Global step or epoch number.
        """
        # Example usage: log_scalars({'lr': 0.001, 'momentum': 0.9}, step)
        for tag, value in tag_value_dict.items():
            full_tag = tag # You can prepend a group tag if needed, e.g., f"Params/{tag}"
            self.log_scalar(full_tag, value, step)

    def log_image(self, tag: str, image: Union[np.ndarray, torch.Tensor], step: int):
        """
        Logs a single image (e.g., a batch of generated images).

        Args:
            tag (str): Data identifier (e.g., 'Generated Images/Epoch').
            image (Union[np.ndarray, torch.Tensor]): The image tensor/array.
                It should be of shape (C, H, W), (H, W, C), or (N, C, H, W).
            step (int): Global step or epoch number.
        """
        # SummaryWriter expects (C, H, W) format
        if isinstance(image, np.ndarray):
             # For numpy images, assume (H, W, C) for simplicity if C=3/1
             if image.ndim == 3 and image.shape[2] in [1, 3]:
                 image = image.transpose((2, 0, 1)) # Convert HWC to CHW
        
        self.writer.add_image(tag, image, step, dataformats='CHW')

    def log_histogram(self, tag: str, values: Union[torch.Tensor, np.ndarray], step: int):
        """
        Logs a histogram of parameter/gradient distributions.

        Args:
            tag (str): Data identifier (e.g., 'Weights/conv1').
            values (Union[torch.Tensor, np.ndarray]): The values to plot as a histogram.
            step (int): Global step or epoch number.
        """
        self.writer.add_histogram(tag, values, step)

    def log_model_graph(self, model: torch.nn.Module, input_to_model: torch.Tensor):
        """
        Logs the model's computational graph to TensorBoard.

        Args:
            model (torch.nn.Module): The PyTorch model instance.
            input_to_model (torch.Tensor): A sample input tensor to trace the graph.
        """
        try:
            self.writer.add_graph(model, input_to_model=input_to_model)
            print("Model graph logged successfully.")
        except Exception as e:
            print(f"Could not log model graph: {e}")

    def log_parameters_and_gradients(self, model: torch.nn.Module, step: int):
        """
        Logs histograms for all model parameters and their gradients.

        Args:
            model (torch.nn.Module): The PyTorch model instance.
            step (int): Global step or epoch number.
        """
        for name, param in model.named_parameters():
            if param.grad is not None:
                # Log parameter distribution
                self.log_histogram(f'Weights/{name}', param.data, step)
                # Log gradient distribution
                self.log_histogram(f'Gradients/{name}', param.grad.data, step)
            else:
                 # Log parameter distribution even if gradients are not computed/available
                 self.log_histogram(f'Weights/{name}', param.data, step)

    def info(self,text:str):
        with open(f'{self.log_path}/{self.time}.txt','+a') as f:
            f.write(text+'\n')
            
    def close(self):
        """
        Closes the SummaryWriter, flushing any pending data.
        """
        self.writer.close()
        print("TensorBoardLogger closed and writer flushed.")

# --- Example Usage (Optional, but highly recommended for a complete file) ---

# def example_training_loop(logger: TensorBoardLogger, epochs: int = 5):
#     """
#     Simulates a simplified PyTorch training loop to demonstrate logger usage.
#     """
#     print("\n--- Starting Example Training Simulation ---")
    
#     # 1. Setup a dummy model and data
#     class SimpleModel(torch.nn.Module):
#         def __init__(self):
#             super().__init__()
#             self.linear = torch.nn.Linear(10, 1)
#         def forward(self, x):
#             return self.linear(x)

#     model = SimpleModel()
#     dummy_input = torch.randn(1, 10)
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#     # 2. Log the model graph once at the beginning
#     logger.log_model_graph(model, dummy_input)

#     # 3. Simulate Epochs
#     global_step = 0
#     for epoch in range(1, epochs + 1):
#         # Simulate data loading
#         dummy_data = torch.randn(32, 10)
#         dummy_labels = torch.randn(32, 1)

#         # Simulate a single training step
#         optimizer.zero_grad()
#         output = model(dummy_data)
        
#         # Simulate a loss calculation (e.g., MSE)
#         loss = (output - dummy_labels).pow(2).mean() 
#         loss.backward()
#         optimizer.step()
        
#         global_step += 1 # Update step after a training batch

#         # Log training metrics
#         logger.log_scalar("Loss/train", loss, global_step)
#         logger.log_scalars({
#             "Params/learning_rate": optimizer.param_groups[0]['lr'],
#             "Params/momentum": 0.9, # Example fixed param
#         }, global_step)

#         # Log histograms of weights and gradients every 10 steps or at end of epoch
#         if epoch % 1 == 0:
#             logger.log_parameters_and_gradients(model, global_step)
            
#         print(f"Epoch {epoch}/{epochs} | Loss: {loss.item():.4f}")

#     # 4. Simulate Validation/Test logging (e.g., at end of training)
#     val_accuracy = 0.85
#     logger.log_scalar("Accuracy/validation", val_accuracy, global_step)
    
#     # 5. Log an example image (e.g., a sample from the test set)
#     # Create a dummy image: C=3, H=64, W=64
#     dummy_image = torch.rand(3, 64, 64) 
#     logger.log_image("Validation/Sample Image", dummy_image, global_step)

#     print("--- Example Training Simulation Finished ---")


# if __name__ == "__main__":
    # Define a unique log directory
    # LOG_DIR = "./runs"
    # EXPERIMENT = "my_pytorch_run"
    
    # # 1. Initialize the logger
    # logger = TensorBoardLogger(log_dir=LOG_DIR, experiment_name=EXPERIMENT)

    # # 2. Run the simulation
    # example_training_loop(logger)

    # # 3. Clean up
    # logger.close()

    # # --- Instructions to view results ---
    # print("\nTo view the results, open your terminal and run the following command:")
    # print(f"tensorboard --logdir={LOG_DIR}")
    # print("Then open your web browser and navigate to the displayed URL (usually http://localhost:6006/)")