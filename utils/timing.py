#%%
import torch
import time

def evaluate_model_inference_time(model_class, checkpoint_path, cfg, input_shape, num_repeats=100, use_cuda=True):
    """
    Evaluate the average inference time for a model loaded from a checkpoint.

    Parameters:
        model_class (torch.nn.Module): The class of the model to load.
        checkpoint_path (str): Path to the model checkpoint.
        cfg (dict): Configuration dictionary for the model.
        input_shape (tuple): Shape of the input tensor (batch_size, channels, sequence_length).
        num_repeats (int): Number of repetitions for inference to calculate average time.
        use_cuda (bool): Whether to use CUDA if available.

    Returns:
        float: Average inference time in seconds.
    """
    # Load the model from checkpoint
    model = model_class.load_from_checkpoint(checkpoint_path, cfg=cfg)
    model.eval()  # Set the model to evaluation mode

    # Create sample input
    sample_input = torch.randn(*input_shape)

    # Warm-up for GPU
    if use_cuda and torch.cuda.is_available():
        model.to("cuda")
        sample_input = sample_input.to("cuda")
        _ = model(sample_input)

    # Measure inference time
    start_time = time.time()

    with torch.no_grad():
        for _ in range(num_repeats):
            _ = model(sample_input)

    end_time = time.time()

    # Calculate average inference time
    avg_inference_time = (end_time - start_time) / num_repeats

    return avg_inference_time

if __name__ == "__main__":
    from models2 import EEGOscillationDetector
    from utils.helpers import load_config

    # Load the configuration
    cfg = load_config("config.yaml")

    # Example usage:
    best_model_path = "path_to_best_model_checkpoint.ckpt"
    avg_time = evaluate_model_inference_time(
        model_class=EEGOscillationDetector,
        checkpoint_path=best_model_path,
        cfg=cfg,
        input_shape=(32, 1, 150),
        num_repeats=100
    )
    print(f"Average Inference Time: {avg_time:.6f} seconds")


#%%