import requests
import json
import time
from pathlib import Path

# API基础URL
BASE_URL = "http://127.0.0.1:8000"

def test_health():
    """测试健康检查"""
    print("=== 测试健康检查 ===")
    response = requests.get(f"{BASE_URL}/health")
    print(f"状态码: {response.status_code}")
    print(f"响应: {response.json()}")
    print()

def test_upload_audio():
    """测试音频上传"""
    print("=== 测试音频上传 ===")
    
    # 检查是否有测试音频文件
    test_audio_path = Path("test_data/input.wav")
    if not test_audio_path.exists():
        print("测试音频文件不存在，跳过上传测试")
        return None
    
    with open(test_audio_path, "rb") as f:
        files = {"file": ("input.wav", f, "audio/wav")}
        response = requests.post(f"{BASE_URL}/upload_audio", files=files)
    
    print(f"状态码: {response.status_code}")
    print(f"响应: {response.json()}")
    print()
    
    if response.status_code == 200:
        return response.json()["file_path"]
    return None

def test_tts(audio_prompt_path):
    """测试标准TTS"""
    print("=== 测试标准TTS ===")
    
    data = {
        "text": "你好，这是一个测试。",
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
    
    response = requests.post(f"{BASE_URL}/tts", json=data)
    print(f"状态码: {response.status_code}")
    print(f"响应: {response.json()}")
    print()
    
    return response.json() if response.status_code == 200 else None

def test_tts_fast(audio_prompt_path):
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
    
    response = requests.post(f"{BASE_URL}/tts_fast", json=data)
    print(f"状态码: {response.status_code}")
    print(f"响应: {response.json()}")
    print()
    
    return response.json() if response.status_code == 200 else None

def test_batch_tts(audio_prompt_path):
    """测试批量TTS"""
    print("=== 测试批量TTS ===")
    
    data = {
        "requests": [
            {
                "text": "第一个测试句子。",
                "audio_prompt_path": audio_prompt_path,
                "verbose": False
            },
            {
                "text": "第二个测试句子。",
                "audio_prompt_path": audio_prompt_path,
                "verbose": False
            }
        ]
    }
    
    response = requests.post(f"{BASE_URL}/batch_tts", json=data)
    print(f"状态码: {response.status_code}")
    print(f"响应: {response.json()}")
    print()
    
    return response.json() if response.status_code == 200 else None

def test_list_files():
    """测试文件列表"""
    print("=== 测试文件列表 ===")
    response = requests.get(f"{BASE_URL}/list_audio_files")
    print(f"状态码: {response.status_code}")
    print(f"响应: {response.json()}")
    print()

def test_download(filename):
    """测试文件下载"""
    print(f"=== 测试文件下载: {filename} ===")
    response = requests.get(f"{BASE_URL}/download/{filename}")
    print(f"状态码: {response.status_code}")
    if response.status_code == 200:
        # 保存下载的文件
        download_path = f"downloaded_{filename}"
        with open(download_path, "wb") as f:
            f.write(response.content)
        print(f"文件已下载到: {download_path}")
    else:
        print(f"下载失败: {response.text}")
    print()

def main():
    """主测试函数"""
    print("开始API测试...")
    print("=" * 50)
    
    # 等待服务启动
    print("等待服务启动...")
    time.sleep(2)
    
    # 测试健康检查
    test_health()
    
    # 测试音频上传
    audio_path = test_upload_audio()
    
    if audio_path:
        # 测试标准TTS
        tts_result = test_tts(audio_path)
        
        # 测试快速TTS
        fast_tts_result = test_tts_fast(audio_path)
        
        # 测试批量TTS
        batch_result = test_batch_tts(audio_path)
        
        # 测试文件列表
        test_list_files()
        
        # 如果有生成的文件，测试下载
        if tts_result and "output_path" in tts_result:
            filename = Path(tts_result["output_path"]).name
            test_download(filename)
    else:
        print("无法获取音频文件路径，跳过TTS测试")
    
    print("API测试完成！")

if __name__ == "__main__":
    main() 