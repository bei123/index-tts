"""
IndexTTS FastAPI 配置文件
"""

import os
from pathlib import Path

# 基础配置
BASE_DIR = Path(__file__).parent
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
OUTPUTS_DIR = BASE_DIR / "outputs"
UPLOADS_DIR = BASE_DIR / "uploads"

# 服务器配置
HOST = os.getenv("API_HOST", "0.0.0.0")
PORT = int(os.getenv("API_PORT", "8000"))
DEBUG = os.getenv("API_DEBUG", "false").lower() == "true"
RELOAD = os.getenv("API_RELOAD", "true").lower() == "true"

# 模型配置 - 从配置文件读取正确的文件名
def get_model_config():
    """从配置文件读取模型配置"""
    config_path = CHECKPOINTS_DIR / "config.yaml"
    if config_path.exists():
        try:
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            return {
                "cfg_path": str(config_path),
                "model_dir": str(CHECKPOINTS_DIR),
                "is_fp16": os.getenv("USE_FP16", "true").lower() == "true",
                "use_cuda_kernel": os.getenv("USE_CUDA_KERNEL", "false").lower() == "true"
            }
        except Exception as e:
            print(f"警告: 读取配置文件失败，使用默认配置: {e}")
    
    # 默认配置
    return {
        "cfg_path": str(CHECKPOINTS_DIR / "config.yaml"),
        "model_dir": str(CHECKPOINTS_DIR),
        "is_fp16": os.getenv("USE_FP16", "true").lower() == "true",
        "use_cuda_kernel": os.getenv("USE_CUDA_KERNEL", "false").lower() == "true"
    }

MODEL_CONFIG = get_model_config()

# 文件配置
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", "50")) * 1024 * 1024  # 50MB
ALLOWED_AUDIO_FORMATS = ['.wav', '.mp3', '.flac', '.m4a']
MAX_UPLOAD_FILES = int(os.getenv("MAX_UPLOAD_FILES", "10"))

# TTS默认参数
DEFAULT_TTS_PARAMS = {
    "max_text_tokens_per_sentence": int(os.getenv("MAX_TEXT_TOKENS", "120")),
    "do_sample": True,
    "top_p": float(os.getenv("TOP_P", "0.8")),
    "top_k": int(os.getenv("TOP_K", "30")),
    "temperature": float(os.getenv("TEMPERATURE", "1.0")),
    "length_penalty": float(os.getenv("LENGTH_PENALTY", "0.0")),
    "num_beams": int(os.getenv("NUM_BEAMS", "3")),
    "repetition_penalty": float(os.getenv("REPETITION_PENALTY", "10.0")),
    "max_mel_tokens": int(os.getenv("MAX_MEL_TOKENS", "600"))
}

# 快速TTS默认参数
DEFAULT_FAST_TTS_PARAMS = {
    "max_text_tokens_per_sentence": int(os.getenv("FAST_MAX_TEXT_TOKENS", "100")),
    "sentences_bucket_max_size": int(os.getenv("SENTENCES_BUCKET_SIZE", "4")),
    **DEFAULT_TTS_PARAMS
}

# CORS配置
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
CORS_ALLOW_CREDENTIALS = True
CORS_ALLOW_METHODS = ["*"]
CORS_ALLOW_HEADERS = ["*"]

# 日志配置
LOG_LEVEL = os.getenv("LOG_LEVEL", "info")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# 缓存配置
ENABLE_CACHE = os.getenv("ENABLE_CACHE", "true").lower() == "true"
CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))  # 1小时

# 安全配置
API_KEY_HEADER = "X-API-Key"
REQUIRE_API_KEY = os.getenv("REQUIRE_API_KEY", "false").lower() == "true"
API_KEYS = os.getenv("API_KEYS", "").split(",") if os.getenv("API_KEYS") else []

# 性能配置
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "10"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "300"))  # 5分钟

# 清理配置
AUTO_CLEANUP = os.getenv("AUTO_CLEANUP", "false").lower() == "true"
CLEANUP_INTERVAL = int(os.getenv("CLEANUP_INTERVAL", "3600"))  # 1小时
MAX_FILE_AGE = int(os.getenv("MAX_FILE_AGE", "86400"))  # 24小时

def get_config():
    """获取配置字典"""
    return {
        "host": HOST,
        "port": PORT,
        "debug": DEBUG,
        "reload": RELOAD,
        "model_config": MODEL_CONFIG,
        "max_file_size": MAX_FILE_SIZE,
        "allowed_audio_formats": ALLOWED_AUDIO_FORMATS,
        "max_upload_files": MAX_UPLOAD_FILES,
        "default_tts_params": DEFAULT_TTS_PARAMS,
        "default_fast_tts_params": DEFAULT_FAST_TTS_PARAMS,
        "cors_origins": CORS_ORIGINS,
        "cors_allow_credentials": CORS_ALLOW_CREDENTIALS,
        "cors_allow_methods": CORS_ALLOW_METHODS,
        "cors_allow_headers": CORS_ALLOW_HEADERS,
        "log_level": LOG_LEVEL,
        "log_format": LOG_FORMAT,
        "enable_cache": ENABLE_CACHE,
        "cache_ttl": CACHE_TTL,
        "require_api_key": REQUIRE_API_KEY,
        "api_keys": API_KEYS,
        "max_concurrent_requests": MAX_CONCURRENT_REQUESTS,
        "request_timeout": REQUEST_TIMEOUT,
        "auto_cleanup": AUTO_CLEANUP,
        "cleanup_interval": CLEANUP_INTERVAL,
        "max_file_age": MAX_FILE_AGE
    }

def print_config():
    """打印配置信息"""
    config = get_config()
    print("IndexTTS API 配置:")
    print("=" * 50)
    for key, value in config.items():
        print(f"{key}: {value}")
    print("=" * 50)

if __name__ == "__main__":
    print_config() 