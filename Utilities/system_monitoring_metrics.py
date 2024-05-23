import psutil
import GPUtil
import wmi
import csv
import time
from datetime import datetime
import pynvml

# Initialize WMI and NVML
w = wmi.WMI()
pynvml.nvmlInit()

def get_system_metrics():
    cpu_usage = psutil.cpu_percent(interval=1)
    ram = psutil.virtual_memory()
    ram_usage_percent = ram.percent
    ram_usage_mb = ram.used / (1024 * 1024)  # Convert bytes to MB
    disk_usage = psutil.disk_usage('/').percent
    gpus = GPUtil.getGPUs()
    gpu_usage = gpus[0].load * 100 if gpus else None
    gpu_temp = gpus[0].temperature if gpus else None
    
    # Get CPU frequency using psutil
    cpu_freq = psutil.cpu_freq().current  # Current CPU frequency in MHz

    # Fetch GPU clock speed (NVIDIA GPUs)
    if gpus:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_clock_speed = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
    else:
        gpu_clock_speed = None

    return {
        'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'cpu_usage': cpu_usage,
        'ram_usage_percent': ram_usage_percent,
        'ram_usage_mb': ram_usage_mb,
        'disk_usage': disk_usage,
        'gpu_usage': gpu_usage,
        'gpu_temp': gpu_temp,
        'cpu_freq_mhz': cpu_freq,  # Now getting from psutil in MHz
        'gpu_clock_speed_mhz': gpu_clock_speed  # in MHz
    }

# Logging and loop setup remains the same
def log_metrics(filename='system_metrics.csv'):
    fieldnames = ['time', 'cpu_usage', 'ram_usage_percent', 'ram_usage_mb', 'disk_usage', 'gpu_usage', 'gpu_temp', 'cpu_freq_mhz', 'gpu_clock_speed_mhz']
    with open(filename, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if file.tell() == 0:
            writer.writeheader()  # Write header only once
        writer.writerow(get_system_metrics())
        
interval = 10  # seconds
try:
    while True:
        log_metrics()
        time.sleep(interval)
except KeyboardInterrupt:
    print("Stopped monitoring.")
