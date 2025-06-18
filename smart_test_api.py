#!/usr/bin/env python3
"""
智能IndexTTS API测试脚本
"""

import requests
import json
import time
import socket
from pathlib import Path

def find_api_service():
    """查找API服务"""
    print("🔍 查找API服务...")
    
    # 尝试常用端口
    ports_to_try = [8000, 8001, 8002, 8003, 8004, 8005, 8080, 3000]
    
    for port in ports_to_try:
        try:
            url = f"http://127.0.0.1:{port}/health"
            response = requests.get(url, timeout=3)
            if response.status_code == 200:
                print(f"✅ 找到API服务: http://127.0.0.1:{port}")
                return f"http://127.0.0.1:{port}"
        except requests.exceptions.RequestException:
            continue
    
    print("❌ 未找到API服务")
    print("请确保API服务已启动")
    return None

def test_health(base_url):
    """测试健康检查"""
    print("=== 测试健康检查 ===")
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        print(f"状态码: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"服务状态: {data.get('status', 'unknown')}")
            print(f"模型加载: {data.get('model_loaded', False)}")
            print(f"设备: {data.get('device', 'unknown')}")
            print(f"运行时间: {data.get('uptime', 0):.1f}秒")
            if data.get('memory_usage'):
                mem = data['memory_usage']
                print(f"内存使用: {mem.get('rss', 0) // (1024*1024)}MB")
        else:
            print(f"响应: {response.text}")
        print()
        return True
    except Exception as e:
        print(f"健康检查失败: {e}")
        print()
        return False

def test_stats(base_url):
    """测试统计信息"""
    print("=== 测试统计信息 ===")
    try:
        response = requests.get(f"{base_url}/stats", timeout=10)
        print(f"状态码: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"总请求数: {data.get('total_requests', 0)}")
            print(f"成功请求: {data.get('successful_requests', 0)}")
            print(f"失败请求: {data.get('failed_requests', 0)}")
            print(f"平均响应时间: {data.get('average_response_time', 0):.3f}秒")
            print(f"音频文件数: {data.get('total_audio_files', 0)}")
        else:
            print(f"响应: {response.text}")
        print()
        return True
    except Exception as e:
        print(f"统计信息获取失败: {e}")
        print()
        return False

def test_upload_audio(base_url):
    """测试音频上传"""
    print("=== 测试音频上传 ===")
    
    # 检查是否有测试音频文件
    test_audio_path = Path("test_data/test.wav")
    if not test_audio_path.exists():
        print("测试音频文件不存在，跳过上传测试")
        print("请确保 test_data/input.wav 文件存在")
        return None
    
    try:
        with open(test_audio_path, "rb") as f:
            files = {"file": ("input.wav", f, "audio/wav")}
            response = requests.post(f"{base_url}/upload_audio", files=files, timeout=30)
        
        print(f"状态码: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"上传成功: {data.get('filename', 'unknown')}")
            print(f"文件路径: {data.get('file_path', 'unknown')}")
            print(f"文件大小: {data.get('file_size', 0)} 字节")
            return data.get("file_path")
        else:
            print(f"上传失败: {response.text}")
            return None
    except Exception as e:
        print(f"音频上传失败: {e}")
        return None

def test_tts(base_url, audio_prompt_path):
    """测试标准TTS"""
    print("=== 测试标准TTS ===")
    
    data = {
        "text": "你好，这是一个标准TTS测试。",
        "audio_prompt_path": audio_prompt_path,
        "verbose": False,
        "max_text_tokens_per_sentence": 120,
        "do_sample": True,
        "top_p": 0.8,
        "top_k": 30,
        "temperature": 1.0,
        "length_penalty": 0.0,
        "num_beams": 3,
        "repetition_penalty": 10.0,
        "max_mel_tokens": 600
    }
    
    try:
        response = requests.post(f"{base_url}/tts", json=data, timeout=300)
        print(f"状态码: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"推理成功: {result.get('inference_time', 0):.2f}秒")
            print(f"输出文件: {result.get('output_path', 'unknown')}")
            return result
        else:
            print(f"TTS失败: {response.text}")
            return None
    except Exception as e:
        print(f"标准TTS测试失败: {e}")
        return None

def test_tts_fast(base_url, audio_prompt_path):
    """测试快速TTS"""
    print("=== 测试快速TTS ===")
    
    data = {
        "text": "这是一个快速TTS测试，应该比标准模式更快。",
        "audio_prompt_path": audio_prompt_path,
        "verbose": False,
        "max_text_tokens_per_sentence": 100,
        "sentences_bucket_max_size": 4,
        "do_sample": True,
        "top_p": 0.8,
        "top_k": 30,
        "temperature": 1.0,
        "length_penalty": 0.0,
        "num_beams": 3,
        "repetition_penalty": 10.0,
        "max_mel_tokens": 600
    }
    
    try:
        response = requests.post(f"{base_url}/tts_fast", json=data, timeout=300)
        print(f"状态码: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"快速推理成功: {result.get('inference_time', 0):.2f}秒")
            print(f"输出文件: {result.get('output_path', 'unknown')}")
            return result
        else:
            print(f"快速TTS失败: {response.text}")
            return None
    except Exception as e:
        print(f"快速TTS测试失败: {e}")
        return None

def test_list_files(base_url):
    """测试文件列表"""
    print("=== 测试文件列表 ===")
    try:
        response = requests.get(f"{base_url}/list_audio_files", timeout=10)
        print(f"状态码: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            files = data.get('files', [])
            print(f"找到 {len(files)} 个音频文件")
            for i, file_info in enumerate(files[:3]):  # 只显示前3个
                print(f"  {i+1}. {file_info.get('filename', 'unknown')} ({file_info.get('created_time_str', 'unknown')})")
        else:
            print(f"获取文件列表失败: {response.text}")
    except Exception as e:
        print(f"文件列表测试失败: {e}")

def main():
    """主测试函数"""
    print("IndexTTS API 智能测试")
    print("=" * 50)
    
    # 查找API服务
    base_url = find_api_service()
    if not base_url:
        print("请先启动API服务:")
        print("  python run_api.py")
        print("  python start_api.py")
        print("  python -m uvicorn api:app --host 127.0.0.1 --port 8000 --reload")
        return
    
    print(f"使用API地址: {base_url}")
    print("=" * 50)
    
    # 等待服务完全启动
    print("等待服务完全启动...")
    time.sleep(3)
    
    # 基础功能测试
    if not test_health(base_url):
        print("❌ 健康检查失败，服务可能未完全启动")
        return
    
    test_stats(base_url)
    
    # 音频上传测试
    audio_path = test_upload_audio(base_url)
    
    if audio_path:
        print("✅ 音频上传成功，开始TTS测试")
        print("-" * 30)
        
        # TTS测试
        tts_result = test_tts(base_url, audio_path)
        fast_tts_result = test_tts_fast(base_url, audio_path)
        
        # 文件列表测试
        test_list_files(base_url)
        
        print("=" * 50)
        print("测试总结:")
        print(f"标准TTS: {'✅ 成功' if tts_result else '❌ 失败'}")
        print(f"快速TTS: {'✅ 成功' if fast_tts_result else '❌ 失败'}")
        
        if tts_result or fast_tts_result:
            print("\n🎉 API测试基本完成！")
            print(f"📚 API文档: {base_url}/docs")
        else:
            print("\n⚠️  TTS功能测试失败，请检查模型和配置")
    else:
        print("❌ 音频上传失败，跳过TTS测试")
        print("请确保 test_data/input.wav 文件存在")

if __name__ == "__main__":
    main() 