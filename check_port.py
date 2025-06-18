#!/usr/bin/env python3
"""
端口检查脚本
"""

import socket
import subprocess
import sys

def check_port(port):
    """检查端口是否被占用"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('localhost', port))
            return True
    except OSError:
        return False

def find_process_using_port(port):
    """查找占用端口的进程"""
    try:
        # Windows系统使用netstat
        result = subprocess.run(
            ['netstat', '-ano'], 
            capture_output=True, 
            text=True, 
            shell=True
        )
        
        lines = result.stdout.split('\n')
        for line in lines:
            if f':{port}' in line and 'LISTENING' in line:
                parts = line.split()
                if len(parts) >= 5:
                    pid = parts[-1]
                    return pid
        return None
    except Exception:
        return None

def main():
    """主函数"""
    print("端口检查工具")
    print("=" * 30)
    
    # 检查常用端口
    ports_to_check = [8000, 8001, 8002, 8003, 8004, 8005]
    
    print("端口状态检查:")
    for port in ports_to_check:
        if check_port(port):
            print(f"✅ 端口 {port}: 可用")
        else:
            print(f"❌ 端口 {port}: 被占用")
            pid = find_process_using_port(port)
            if pid:
                print(f"   占用进程PID: {pid}")
    
    print("\n建议:")
    print("- 如果端口8000被占用，可以尝试其他端口")
    print("- 或者使用以下命令终止占用进程:")
    print("  taskkill /PID <进程ID> /F")
    print("\n启动API服务:")
    print("python run_api.py")

if __name__ == "__main__":
    main() 