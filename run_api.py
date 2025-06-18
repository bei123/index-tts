#!/usr/bin/env python3
"""
IndexTTS API 简单启动脚本
"""

import uvicorn
import socket
from pathlib import Path

def find_available_port(start_port=8000, max_attempts=10):
    """查找可用端口"""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return port
        except OSError:
            continue
    return None

def main():
    """主函数"""
    print("IndexTTS API 服务启动中...")
    print("=" * 40)
    
    # 检查模型文件
    config_path = Path("checkpoints/config.yaml")
    if not config_path.exists():
        print("❌ 配置文件不存在，请确保模型文件已下载")
        return
    
    # 查找可用端口
    port = find_available_port(8000)
    if port is None:
        print("❌ 无法找到可用端口")
        return
    
    print(f"✅ 使用端口: {port}")
    print(f"🌐 服务地址: http://localhost:{port}")
    print(f"📚 API文档: http://localhost:{port}/docs")
    print("=" * 40)
    
    try:
        # 启动服务
        uvicorn.run(
            "api:app",
            host="127.0.0.1",  # 使用localhost
            port=port,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n👋 服务已停止")
    except Exception as e:
        print(f"❌ 启动失败: {e}")

if __name__ == "__main__":
    main() 