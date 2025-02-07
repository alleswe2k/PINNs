import torch
import time

def benchmark_cpu(N=2000):
    # Create two large random matrices on CPU
    a = torch.randn(N, N)
    b = torch.randn(N, N)

    # Warm-up (optional)
    c = torch.mm(a, b)

    start = time.time()
    # Matrix multiplication on CPU
    c = torch.mm(a, b)
    end = time.time()
    
    print(f"CPU matrix multiplication time: {end - start:.4f} seconds")

def benchmark_gpu(N=2000):
    if not torch.cuda.is_available():
        print("CUDA is not available on this machine.")
        return

    device = torch.device("cuda")
    # Create matrices directly on the GPU
    a = torch.randn(N, N, device=device)
    b = torch.randn(N, N, device=device)

    # Warm-up: run a dummy operation to initialize the GPU and CUDA context
    c = torch.mm(a, b)
    torch.cuda.synchronize()  # wait for warm-up to complete

    start = time.time()
    c = torch.mm(a, b)
    torch.cuda.synchronize()  # wait for GPU operations to complete
    end = time.time()
    
    print(f"GPU matrix multiplication time: {end - start:.4f} seconds")

if __name__ == '__main__':
    matrix_size = 10000  # You can adjust this size; larger matrices will emphasize GPU speed-up.
    print("Benchmarking on CPU...")
    benchmark_cpu(matrix_size)
    print("Benchmarking on GPU...")
    benchmark_gpu(matrix_size)
