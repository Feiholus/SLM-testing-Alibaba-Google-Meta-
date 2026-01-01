# Local SLM Benchmarking: Meta vs. Alibaba vs. Google

This project focuses on the local deployment and performance analysis of Small Language Models (SLMs) from three different global providers. The primary goal was to implement a programmatic inference solution (no GUI) and compare the efficiency of CPU-based execution versus GPU-accelerated execution using CUDA.

## 📊 Performance Overview (RTX 4050 vs. CPU)

| Model | Provider | GPU Speed (tokens/s) | CPU Speed (tokens/s) | GPU Boost |
| :--- | :--- | :--- | :--- | :--- |
| **Llama-3.2-1B-Instruct** | Meta | **61.24** | 17.73 | ~3.5x |
| **Qwen2.5-1.5B-Instruct** | Alibaba | **61.07** | 8.32 | **7.3x** |
| **Gemma-2-2B-it** | Google | **56.10** | 14.71 | ~3.8x |

## 🛠 Tech Stack
- **Inference Engine:** `llama-cpp-python` (built with CUDA 12.4 support).
- **Quantization:** GGUF format (Q8_0 and Q4_K_M) to ensure all models stay under the 3GB VRAM/RAM limit.
- **Hardware:** 
  - **GPU:** NVIDIA GeForce RTX 4050 Laptop (6GB VRAM).
  - **CPU:** 13th Gen Intel Core.
- **Environment:** Isolated Python `venv`.

## ⚙️ Implementation Details
The implementation is a pure Python CLI script. It handles model loading, resource allocation, and automated benchmarking. 

### Key Features:
- **Zero GUI:** All interactions occur via the terminal to minimize overhead.
- **Dynamic Resource Management:** Using the `n_gpu_layers` parameter to toggle between full GPU offloading (`-1`) and pure CPU execution (`0`).
- **Windows DLL Fix:** Implemented manual DLL directory injection to ensure the Python environment correctly links to the NVIDIA CUDA Toolkit binaries:
  ```python
  import os
  os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin")
