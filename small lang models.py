import os
import sys
import time

# --- БЛОК ИСПРАВЛЕНИЯ DLL (Критически важно для Windows) ---
# Проверь, что путь v12.4 совпадает с тем, что ты установил
cuda_bin_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin"
if os.path.exists(cuda_bin_path):
    os.add_dll_directory(cuda_bin_path)

try:
    from llama_cpp import Llama
    print("✅ Библиотека загружена. Начинаю тест на видеокарте RTX 4050...")
except Exception as e:
    print(f"❌ Ошибка импорта: {e}")
    sys.exit(1)

# Список моделей
models = [
    {"name": "Meta (Llama-3.2-1B)", "file": "Llama-3.2-1B-Instruct-Q8_0.gguf"},
    {"name": "Alibaba (Qwen2.5-1.5B)", "file": "Qwen2.5-1.5B-Instruct-Q8_0.gguf"},
    {"name": "Google (Gemma-2-2B)", "file": "gemma-2-2b-it-Q4_K_M.gguf"}
]

def run_benchmark(model_info, use_gpu=True):
    layers = -1 if use_gpu else 0
    mode = "GPU (RTX 4050)" if use_gpu else "CPU"
    
    print(f"\n" + "="*50)
    print(f"ТЕСТ МОДЕЛИ: {model_info['name']} | РЕЖИМ: {mode}")
    print("="*50)
    
    if not os.path.exists(model_info['file']):
        print(f"Ошибка: Файл {model_info['file']} не найден в папке Documents!")
        return

    try:
        # Инициализация
        llm = Llama(
            model_path=model_info['file'],
            n_gpu_layers=layers, 
            n_ctx=1024,
            verbose=False
        )

        # Тестовый запрос
        start = time.time()
        output = llm.create_chat_completion(
            messages=[{"role": "user", "content": "Write a short slogan for a tech company."}],
            max_tokens=50
        )
        end = time.time()

        # Расчет результатов
        text = output["choices"][0]["message"]["content"].strip()
        tokens = output["usage"]["completion_tokens"]
        tps = tokens / (end - start)

        print(f"ОТВЕТ: {text}")
        print(f"СКОРОСТЬ: {tps:.2f} токенов в секунду")
        
        del llm 
    except Exception as e:
        print(f"Ошибка запуска: {e}")

if __name__ == "__main__":
    for m in models:
        # Сначала на GPU, потом (для сравнения) на CPU
        run_benchmark(m, use_gpu=True)
        run_benchmark(m, use_gpu=False)