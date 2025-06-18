# IndexTTS FastAPI 服务

基于IndexTTS的FastAPI文本转语音API服务，提供RESTful接口进行文本转语音转换。

## 功能特性

- 🚀 **标准TTS**: 支持单次文本转语音
- ⚡ **快速TTS**: 批量处理优化，速度提升2-10倍
- 📦 **批量处理**: 支持批量文本转语音
- 📁 **文件管理**: 音频文件上传、下载、列表、删除
- 🔧 **参数调优**: 支持多种TTS参数调整
- 📊 **健康监控**: 服务状态和模型加载状态检查

## 安装依赖

```bash
pip install -r requirements_api.txt
```

## 启动服务

```bash
python api.py
```

服务将在 `http://localhost:8000` 启动，API文档可在 `http://localhost:8000/docs` 查看。

## API接口

### 1. 健康检查

**GET** `/health`

检查服务状态和模型加载情况。

```bash
curl http://localhost:8000/health
```

响应示例：
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda:0"
}
```

### 2. 手动初始化模型

**POST** `/init_model`

手动重新初始化TTS模型。

```bash
curl -X POST http://localhost:8000/init_model
```

### 3. 音频文件上传

**POST** `/upload_audio`

上传参考音频文件。

```bash
curl -X POST -F "file=@your_audio.wav" http://localhost:8000/upload_audio
```

支持格式：wav, mp3, flac, m4a

### 4. 标准文本转语音

**POST** `/tts`

执行标准TTS推理。

```bash
curl -X POST http://localhost:8000/tts \
  -H "Content-Type: application/json" \
  -d '{
    "text": "你好，这是一个测试。",
    "audio_prompt_path": "/path/to/reference.wav",
    "verbose": false,
    "max_text_tokens_per_sentence": 120,
    "do_sample": true,
    "top_p": 0.8,
    "top_k": 30,
    "temperature": 1.0,
    "length_penalty": 0.0,
    "num_beams": 3,
    "repetition_penalty": 10.0,
    "max_mel_tokens": 600
  }'
```

### 5. 快速文本转语音

**POST** `/tts_fast`

执行快速TTS推理（批量处理优化）。

```bash
curl -X POST http://localhost:8000/tts_fast \
  -H "Content-Type: application/json" \
  -d '{
    "text": "这是一个快速TTS测试。",
    "audio_prompt_path": "/path/to/reference.wav",
    "verbose": false,
    "max_text_tokens_per_sentence": 100,
    "sentences_bucket_max_size": 4,
    "do_sample": true,
    "top_p": 0.8,
    "top_k": 30,
    "temperature": 1.0,
    "length_penalty": 0.0,
    "num_beams": 3,
    "repetition_penalty": 10.0,
    "max_mel_tokens": 600
  }'
```

### 6. 批量文本转语音

**POST** `/batch_tts`

批量处理多个TTS请求。

```bash
curl -X POST http://localhost:8000/batch_tts \
  -H "Content-Type: application/json" \
  -d '{
    "requests": [
      {
        "text": "第一个句子。",
        "audio_prompt_path": "/path/to/reference.wav",
        "verbose": false
      },
      {
        "text": "第二个句子。",
        "audio_prompt_path": "/path/to/reference.wav",
        "verbose": false
      }
    ]
  }'
```

### 7. 文件管理

#### 列出音频文件

**GET** `/list_audio_files`

```bash
curl http://localhost:8000/list_audio_files
```

#### 下载音频文件

**GET** `/download/{filename}`

```bash
curl -O http://localhost:8000/download/your_audio.wav
```

#### 删除音频文件

**DELETE** `/delete_audio/{filename}`

```bash
curl -X DELETE http://localhost:8000/delete_audio/your_audio.wav
```

## 参数说明

### TTS参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `text` | string | - | 要转换的文本内容 |
| `audio_prompt_path` | string | - | 参考音频文件路径 |
| `output_path` | string | 自动生成 | 输出音频文件路径 |
| `verbose` | boolean | false | 是否显示详细日志 |
| `max_text_tokens_per_sentence` | int | 120/100 | 每句话的最大token数 |
| `do_sample` | boolean | true | 是否使用采样 |
| `top_p` | float | 0.8 | top_p采样参数 |
| `top_k` | int | 30 | top_k采样参数 |
| `temperature` | float | 1.0 | 温度参数 |
| `length_penalty` | float | 0.0 | 长度惩罚 |
| `num_beams` | int | 3 | 束搜索数量 |
| `repetition_penalty` | float | 10.0 | 重复惩罚 |
| `max_mel_tokens` | int | 600 | 最大mel token数 |

### 快速TTS特有参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `sentences_bucket_max_size` | int | 4 | 句子分桶最大大小 |

## 使用示例

### Python客户端示例

```python
import requests

# 健康检查
response = requests.get("http://localhost:8000/health")
print(response.json())

# 上传音频文件
with open("reference.wav", "rb") as f:
    files = {"file": ("reference.wav", f, "audio/wav")}
    response = requests.post("http://localhost:8000/upload_audio", files=files)
    audio_path = response.json()["file_path"]

# 执行TTS
data = {
    "text": "你好，这是一个测试。",
    "audio_prompt_path": audio_path,
    "verbose": False
}
response = requests.post("http://localhost:8000/tts", json=data)
result = response.json()

# 下载生成的音频
if result["success"]:
    output_filename = result["output_path"].split("/")[-1]
    response = requests.get(f"http://localhost:8000/download/{output_filename}")
    with open(f"output_{output_filename}", "wb") as f:
        f.write(response.content)
```

### JavaScript客户端示例

```javascript
// 健康检查
fetch('http://localhost:8000/health')
  .then(response => response.json())
  .then(data => console.log(data));

// 上传音频文件
const formData = new FormData();
formData.append('file', audioFile);

fetch('http://localhost:8000/upload_audio', {
  method: 'POST',
  body: formData
})
.then(response => response.json())
.then(data => {
  // 执行TTS
  return fetch('http://localhost:8000/tts', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      text: '你好，这是一个测试。',
      audio_prompt_path: data.file_path,
      verbose: false
    })
  });
})
.then(response => response.json())
.then(result => console.log(result));
```

## 测试

运行测试脚本：

```bash
python test_api.py
```

## 目录结构

```
.
├── api.py                 # FastAPI主服务文件
├── test_api.py           # API测试脚本
├── requirements_api.txt  # API依赖文件
├── README_API.md        # API使用说明
├── outputs/             # 生成的音频文件目录
├── uploads/             # 上传的音频文件目录
└── checkpoints/         # 模型文件目录
```

## 注意事项

1. **模型文件**: 确保 `checkpoints/` 目录包含必要的模型文件
2. **GPU支持**: 服务会自动检测并使用可用的GPU
3. **内存管理**: 长时间运行建议定期重启服务释放内存
4. **文件清理**: 定期清理 `outputs/` 和 `uploads/` 目录
5. **并发处理**: 服务支持并发请求，但建议控制并发数量

## 故障排除

### 常见问题

1. **模型初始化失败**
   - 检查模型文件是否存在
   - 确认CUDA环境配置
   - 查看服务日志

2. **内存不足**
   - 减少 `sentences_bucket_max_size` 参数
   - 降低 `max_text_tokens_per_sentence` 参数
   - 重启服务释放内存

3. **音频文件格式不支持**
   - 确保音频文件格式为 wav, mp3, flac, m4a
   - 检查文件是否损坏

4. **推理速度慢**
   - 使用快速TTS模式 (`/tts_fast`)
   - 调整 `sentences_bucket_max_size` 参数
   - 检查GPU使用情况

## 许可证

本项目遵循原IndexTTS项目的许可证。 