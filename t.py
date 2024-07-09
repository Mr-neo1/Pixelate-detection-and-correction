import torch

def check_gpu():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu = torch.cuda.get_device_name(0)
        cuda_version = torch.version.cuda
        print(f"Using GPU: {gpu}")
        print(f"CUDA Version: {cuda_version}")
    else:
        print("CUDA is not available. Using CPU.")

if __name__ == "__main__":
    check_gpu()
