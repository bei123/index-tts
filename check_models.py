#!/usr/bin/env python3
"""
IndexTTS 模型文件检查脚本
"""

import os
import yaml
from pathlib import Path

def check_model_files():
    """检查模型文件"""
    print("IndexTTS 模型文件检查")
    print("=" * 50)
    
    checkpoints_dir = Path("checkpoints")
    if not checkpoints_dir.exists():
        print("❌ checkpoints 目录不存在")
        return False
    
    # 检查配置文件
    config_path = checkpoints_dir / "config.yaml"
    if not config_path.exists():
        print("❌ 配置文件不存在: checkpoints/config.yaml")
        return False
    
    print("✅ 配置文件存在")
    
    # 读取配置文件
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        print("✅ 配置文件读取成功")
        print(f"   版本: {config.get('version', 'unknown')}")
        
    except Exception as e:
        print(f"❌ 配置文件读取失败: {e}")
        return False
    
    # 检查模型文件
    model_files = {
        "GPT模型": config.get('gpt_checkpoint', 'gpt.pth'),
        "BigVGAN模型": config.get('bigvgan_checkpoint', 'bigvgan_generator.pth'),
        "DVAE模型": config.get('dvae_checkpoint', 'dvae.pth'),
        "BPE模型": "bpe.model"
    }
    
    all_files_exist = True
    total_size = 0
    
    for model_name, filename in model_files.items():
        file_path = checkpoints_dir / filename
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            total_size += size_mb
            print(f"✅ {model_name}: {filename} ({size_mb:.1f}MB)")
        else:
            print(f"❌ {model_name}: {filename} (文件不存在)")
            all_files_exist = False
    
    print(f"\n📊 模型文件总大小: {total_size:.1f}MB")
    
    # 检查其他文件
    other_files = [
        "unigram_12000.vocab",
        "bigvgan_discriminator.pth"
    ]
    
    print("\n其他文件检查:")
    for filename in other_files:
        file_path = checkpoints_dir / filename
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"✅ {filename} ({size_mb:.1f}MB)")
        else:
            print(f"⚠️  {filename} (可选文件)")
    
    if all_files_exist:
        print("\n🎉 所有必需的模型文件都存在！")
        return True
    else:
        print("\n❌ 缺少必需的模型文件")
        return False

def check_dependencies():
    """检查依赖"""
    print("\n依赖检查:")
    print("-" * 30)
    
    required_packages = [
        'torch',
        'torchaudio',
        'fastapi',
        'uvicorn',
        'pydantic',
        'omegaconf',
        'yaml'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n缺少依赖包: {', '.join(missing_packages)}")
        print("请运行: pip install -r requirements_api.txt")
        return False
    
    print("\n✅ 所有依赖包都已安装")
    return True

def check_gpu():
    """检查GPU"""
    print("\nGPU检查:")
    print("-" * 30)
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ CUDA可用: {gpu_name} (共 {gpu_count} 个GPU)")
            
            # 检查CUDA版本
            cuda_version = torch.version.cuda
            print(f"   CUDA版本: {cuda_version}")
            
            # 检查PyTorch版本
            pytorch_version = torch.__version__
            print(f"   PyTorch版本: {pytorch_version}")
            
            return True
        elif hasattr(torch, "mps") and torch.backends.mps.is_available():
            print("✅ MPS (Apple Silicon) 可用")
            return True
        else:
            print("⚠️  未检测到GPU，将使用CPU模式")
            return False
    except Exception as e:
        print(f"❌ GPU检查失败: {e}")
        return False

def main():
    """主函数"""
    print("IndexTTS 环境检查")
    print("=" * 50)
    
    # 检查模型文件
    models_ok = check_model_files()
    
    # 检查依赖
    deps_ok = check_dependencies()
    
    # 检查GPU
    gpu_ok = check_gpu()
    
    print("\n" + "=" * 50)
    print("检查结果总结:")
    print(f"模型文件: {'✅ 正常' if models_ok else '❌ 异常'}")
    print(f"依赖包: {'✅ 正常' if deps_ok else '❌ 异常'}")
    print(f"GPU环境: {'✅ 正常' if gpu_ok else '⚠️  CPU模式'}")
    
    if models_ok and deps_ok:
        print("\n🎉 环境检查通过！可以启动API服务")
        print("运行命令: python start_api.py")
    else:
        print("\n❌ 环境检查失败，请解决上述问题后重试")

if __name__ == "__main__":
    main() 