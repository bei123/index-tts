#!/usr/bin/env python3
"""
IndexTTS FastAPI 服务启动脚本
"""

import os
import sys
import subprocess
import time
import socket
from pathlib import Path

def check_port_available(port):
    """检查端口是否可用"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('localhost', port))
            return True
    except OSError:
        return False

def find_available_port(start_port=8000, max_attempts=10):
    """查找可用端口"""
    for port in range(start_port, start_port + max_attempts):
        if check_port_available(port):
            return port
    return None

def check_dependencies():
    """检查依赖是否安装"""
    required_packages = [
        'fastapi',
        'uvicorn',
        'pydantic',
        'torch',
        'torchaudio'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ 缺少依赖包: {', '.join(missing_packages)}")
        print("请运行: pip install -r requirements_api.txt")
        return False
    
    print("✅ 依赖检查通过")
    return True

def check_model_files():
    """检查模型文件是否存在"""
    # 从配置文件读取正确的文件名
    config_path = Path("checkpoints/config.yaml")
    if not config_path.exists():
        print("❌ 配置文件不存在: checkpoints/config.yaml")
        return False
    
    try:
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 获取配置文件中定义的文件名
        gpt_checkpoint = config.get('gpt_checkpoint', 'gpt.pth')
        bigvgan_checkpoint = config.get('bigvgan_checkpoint', 'bigvgan_generator.pth')
        dvae_checkpoint = config.get('dvae_checkpoint', 'dvae.pth')
        
        required_files = [
            f"checkpoints/config.yaml",
            f"checkpoints/{gpt_checkpoint}",
            f"checkpoints/{bigvgan_checkpoint}",
            f"checkpoints/{dvae_checkpoint}",
            "checkpoints/bpe.model"
        ]
        
    except Exception as e:
        print(f"❌ 读取配置文件失败: {e}")
        # 使用默认文件名作为备选
        required_files = [
            "checkpoints/config.yaml",
            "checkpoints/gpt.pth",
            "checkpoints/bigvgan_generator.pth",
            "checkpoints/dvae.pth",
            "checkpoints/bpe.model"
        ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ 缺少模型文件:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        print("请确保模型文件已正确放置在 checkpoints/ 目录中")
        return False
    
    print("✅ 模型文件检查通过")
    return True

def create_directories():
    """创建必要的目录"""
    directories = ["outputs", "uploads"]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ 目录 {directory}/ 已创建或存在")

def check_gpu():
    """检查GPU可用性"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ GPU可用: {gpu_name} (共 {gpu_count} 个)")
            return True
        elif hasattr(torch, "mps") and torch.backends.mps.is_available():
            print("✅ MPS (Apple Silicon) 可用")
            return True
        else:
            print("⚠️  未检测到GPU，将使用CPU模式（速度较慢）")
            return False
    except Exception as e:
        print(f"⚠️  GPU检查失败: {e}")
        return False

def start_server():
    """启动API服务器"""
    print("\n🚀 启动IndexTTS FastAPI服务...")
    print("=" * 50)
    
    # 查找可用端口
    port = find_available_port(8000)
    if port is None:
        print("❌ 无法找到可用端口")
        return False
    
    print(f"📡 使用端口: {port}")
    print(f"🌐 服务地址: http://localhost:{port}")
    print(f"📚 API文档: http://localhost:{port}/docs")
    
    try:
        # 启动服务器 - 使用localhost而不是0.0.0.0
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "api:app", 
            "--host", "127.0.0.1",  # 改为localhost
            "--port", str(port), 
            "--reload"
        ], check=True)
    except KeyboardInterrupt:
        print("\n👋 服务已停止")
    except subprocess.CalledProcessError as e:
        print(f"❌ 服务启动失败: {e}")
        return False
    
    return True

def main():
    """主函数"""
    print("IndexTTS FastAPI 服务启动器")
    print("=" * 50)
    
    # 检查依赖
    if not check_dependencies():
        sys.exit(1)
    
    # 检查模型文件
    if not check_model_files():
        sys.exit(1)
    
    # 检查GPU
    check_gpu()
    
    # 创建目录
    create_directories()
    
    # 启动服务器
    start_server()

if __name__ == "__main__":
    main() 