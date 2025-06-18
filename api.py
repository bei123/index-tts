import os
import uuid
import time
import tempfile
import logging
from typing import Optional, List, Dict, Any
from pathlib import Path
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks, Depends, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field
import uvicorn

from indextts.infer import IndexTTS
from config_api import get_config, CORS_ORIGINS, CORS_ALLOW_CREDENTIALS, CORS_ALLOW_METHODS, CORS_ALLOW_HEADERS

# 获取配置
config = get_config()

# 配置日志
logging.basicConfig(
    level=getattr(logging, config["log_level"].upper()),
    format=config["log_format"]
)
logger = logging.getLogger(__name__)

# 全局TTS实例
tts_instance: Optional[IndexTTS] = None

# 请求模型
class TTSRequest(BaseModel):
    text: str = Field(..., description="要转换的文本内容", min_length=1, max_length=10000)
    audio_prompt_path: Optional[str] = Field(None, description="参考音频文件路径")
    output_path: Optional[str] = Field(None, description="输出音频文件路径")
    verbose: bool = Field(False, description="是否显示详细日志")
    max_text_tokens_per_sentence: int = Field(config["default_tts_params"]["max_text_tokens_per_sentence"], description="每句话的最大token数")
    do_sample: bool = Field(config["default_tts_params"]["do_sample"], description="是否使用采样")
    top_p: float = Field(config["default_tts_params"]["top_p"], description="top_p采样参数")
    top_k: int = Field(config["default_tts_params"]["top_k"], description="top_k采样参数")
    temperature: float = Field(config["default_tts_params"]["temperature"], description="温度参数")
    length_penalty: float = Field(config["default_tts_params"]["length_penalty"], description="长度惩罚")
    num_beams: int = Field(config["default_tts_params"]["num_beams"], description="束搜索数量")
    repetition_penalty: float = Field(config["default_tts_params"]["repetition_penalty"], description="重复惩罚")
    max_mel_tokens: int = Field(config["default_tts_params"]["max_mel_tokens"], description="最大mel token数")

class FastTTSRequest(BaseModel):
    text: str = Field(..., description="要转换的文本内容", min_length=1, max_length=10000)
    audio_prompt_path: Optional[str] = Field(None, description="参考音频文件路径")
    output_path: Optional[str] = Field(None, description="输出音频文件路径")
    verbose: bool = Field(False, description="是否显示详细日志")
    max_text_tokens_per_sentence: int = Field(config["default_fast_tts_params"]["max_text_tokens_per_sentence"], description="每句话的最大token数")
    sentences_bucket_max_size: int = Field(config["default_fast_tts_params"]["sentences_bucket_max_size"], description="句子分桶最大大小")
    do_sample: bool = Field(config["default_fast_tts_params"]["do_sample"], description="是否使用采样")
    top_p: float = Field(config["default_fast_tts_params"]["top_p"], description="top_p采样参数")
    top_k: int = Field(config["default_fast_tts_params"]["top_k"], description="top_k采样参数")
    temperature: float = Field(config["default_fast_tts_params"]["temperature"], description="温度参数")
    length_penalty: float = Field(config["default_fast_tts_params"]["length_penalty"], description="长度惩罚")
    num_beams: int = Field(config["default_fast_tts_params"]["num_beams"], description="束搜索数量")
    repetition_penalty: float = Field(config["default_fast_tts_params"]["repetition_penalty"], description="重复惩罚")
    max_mel_tokens: int = Field(config["default_fast_tts_params"]["max_mel_tokens"], description="最大mel token数")

class BatchTTSRequest(BaseModel):
    requests: List[TTSRequest] = Field(..., description="批量TTS请求列表", max_items=100)

class HealthResponse(BaseModel):
    status: str = Field(..., description="服务状态")
    model_loaded: bool = Field(..., description="模型是否已加载")
    device: str = Field(..., description="当前设备")
    uptime: float = Field(..., description="服务运行时间（秒）")
    memory_usage: Optional[Dict[str, Any]] = Field(None, description="内存使用情况")

class StatsResponse(BaseModel):
    total_requests: int = Field(..., description="总请求数")
    successful_requests: int = Field(..., description="成功请求数")
    failed_requests: int = Field(..., description="失败请求数")
    average_response_time: float = Field(..., description="平均响应时间（秒）")
    total_audio_files: int = Field(..., description="生成的音频文件总数")

# 统计信息
stats = {
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "response_times": [],
    "start_time": time.time()
}

# 初始化TTS模型
def init_tts_model():
    """初始化TTS模型"""
    global tts_instance
    try:
        logger.info("正在初始化IndexTTS模型...")
        tts_instance = IndexTTS(**config["model_config"])
        logger.info("IndexTTS模型初始化完成")
        return True
    except Exception as e:
        logger.error(f"模型初始化失败: {e}")
        return False

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化模型
    logger.info("IndexTTS API 服务启动中...")
    init_tts_model()
    logger.info(f"服务启动完成，监听地址: {config['host']}:{config['port']}")
    
    yield
    
    # 关闭时清理资源
    logger.info("IndexTTS API 服务关闭中...")
    global tts_instance
    if tts_instance:
        del tts_instance
        tts_instance = None
    logger.info("服务已关闭")

# 创建FastAPI应用
app = FastAPI(
    title="IndexTTS API",
    description="基于IndexTTS的文本转语音API服务",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=CORS_ALLOW_CREDENTIALS,
    allow_methods=CORS_ALLOW_METHODS,
    allow_headers=CORS_ALLOW_HEADERS,
)

# 添加可信主机中间件（可选）
if config.get("trusted_hosts"):
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=config["trusted_hosts"]
    )

# 依赖函数
def get_tts_instance():
    """获取TTS实例"""
    if tts_instance is None:
        raise HTTPException(status_code=503, detail="TTS模型未初始化，请稍后重试")
    return tts_instance

def update_stats(success: bool, response_time: float):
    """更新统计信息"""
    stats["total_requests"] += 1
    if success:
        stats["successful_requests"] += 1
    else:
        stats["failed_requests"] += 1
    
    stats["response_times"].append(response_time)
    # 只保留最近1000个响应时间
    if len(stats["response_times"]) > 1000:
        stats["response_times"] = stats["response_times"][-1000:]

@app.get("/", response_model=Dict[str, str])
async def root():
    """根路径"""
    return {
        "message": "IndexTTS API 服务正在运行", 
        "docs": "/docs",
        "version": "1.0.0",
        "status": "healthy"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查"""
    import psutil
    
    uptime = time.time() - stats["start_time"]
    memory_info = None
    
    try:
        process = psutil.Process()
        memory_info = {
            "rss": process.memory_info().rss,
            "vms": process.memory_info().vms,
            "percent": process.memory_percent()
        }
    except ImportError:
        pass  # psutil未安装
    
    return HealthResponse(
        status="healthy",
        model_loaded=tts_instance is not None,
        device=tts_instance.device if tts_instance else "unknown",
        uptime=round(uptime, 2),
        memory_usage=memory_info
    )

@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """获取服务统计信息"""
    avg_response_time = 0
    if stats["response_times"]:
        avg_response_time = sum(stats["response_times"]) / len(stats["response_times"])
    
    # 统计音频文件数量
    output_dir = Path("outputs")
    total_audio_files = len(list(output_dir.glob("*.wav"))) if output_dir.exists() else 0
    
    return StatsResponse(
        total_requests=stats["total_requests"],
        successful_requests=stats["successful_requests"],
        failed_requests=stats["failed_requests"],
        average_response_time=round(avg_response_time, 3),
        total_audio_files=total_audio_files
    )

@app.post("/init_model")
async def init_model():
    """手动初始化模型"""
    success = init_tts_model()
    if success:
        return {"message": "模型初始化成功", "device": tts_instance.device}
    else:
        raise HTTPException(status_code=500, detail="模型初始化失败")

@app.post("/tts")
async def text_to_speech(request: TTSRequest, tts: IndexTTS = Depends(get_tts_instance)):
    """标准文本转语音"""
    start_time = time.time()
    success = False
    
    try:
        # 生成输出路径
        if not request.output_path:
            output_dir = Path("outputs")
            output_dir.mkdir(exist_ok=True)
            request.output_path = str(output_dir / f"{uuid.uuid4()}.wav")
        
        # 检查参考音频文件
        if not request.audio_prompt_path or not os.path.exists(request.audio_prompt_path):
            raise HTTPException(status_code=400, detail="参考音频文件不存在")
        
        # 执行TTS推理
        result = tts.infer(
            audio_prompt=request.audio_prompt_path,
            text=request.text,
            output_path=request.output_path,
            verbose=request.verbose,
            max_text_tokens_per_sentence=request.max_text_tokens_per_sentence,
            do_sample=request.do_sample,
            top_p=request.top_p,
            top_k=request.top_k,
            temperature=request.temperature,
            length_penalty=request.length_penalty,
            num_beams=request.num_beams,
            repetition_penalty=request.repetition_penalty,
            max_mel_tokens=request.max_mel_tokens
        )
        
        end_time = time.time()
        success = True
        update_stats(success, end_time - start_time)
        
        return {
            "success": True,
            "output_path": request.output_path,
            "inference_time": round(end_time - start_time, 2),
            "text_length": len(request.text),
            "message": "TTS推理完成"
        }
        
    except Exception as e:
        end_time = time.time()
        update_stats(success, end_time - start_time)
        logger.error(f"TTS推理失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"TTS推理失败: {str(e)}")

@app.post("/tts_fast")
async def text_to_speech_fast(request: FastTTSRequest, tts: IndexTTS = Depends(get_tts_instance)):
    """快速文本转语音（批量处理优化）"""
    start_time = time.time()
    success = False
    
    try:
        # 生成输出路径
        if not request.output_path:
            output_dir = Path("outputs")
            output_dir.mkdir(exist_ok=True)
            request.output_path = str(output_dir / f"{uuid.uuid4()}.wav")
        
        # 检查参考音频文件
        if not request.audio_prompt_path or not os.path.exists(request.audio_prompt_path):
            raise HTTPException(status_code=400, detail="参考音频文件不存在")
        
        # 执行快速TTS推理
        result = tts.infer_fast(
            audio_prompt=request.audio_prompt_path,
            text=request.text,
            output_path=request.output_path,
            verbose=request.verbose,
            max_text_tokens_per_sentence=request.max_text_tokens_per_sentence,
            sentences_bucket_max_size=request.sentences_bucket_max_size,
            do_sample=request.do_sample,
            top_p=request.top_p,
            top_k=request.top_k,
            temperature=request.temperature,
            length_penalty=request.length_penalty,
            num_beams=request.num_beams,
            repetition_penalty=request.repetition_penalty,
            max_mel_tokens=request.max_mel_tokens
        )
        
        end_time = time.time()
        success = True
        update_stats(success, end_time - start_time)
        
        return {
            "success": True,
            "output_path": request.output_path,
            "inference_time": round(end_time - start_time, 2),
            "text_length": len(request.text),
            "message": "快速TTS推理完成"
        }
        
    except Exception as e:
        end_time = time.time()
        update_stats(success, end_time - start_time)
        logger.error(f"快速TTS推理失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"快速TTS推理失败: {str(e)}")

@app.post("/upload_audio")
async def upload_audio(file: UploadFile = File(...)):
    """上传参考音频文件"""
    start_time = time.time()
    success = False
    
    try:
        # 检查文件类型
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in config["allowed_audio_formats"]:
            raise HTTPException(
                status_code=400, 
                detail=f"不支持的文件格式。支持格式: {', '.join(config['allowed_audio_formats'])}"
            )
        
        # 检查文件大小
        if file.size and file.size > config["max_file_size"]:
            raise HTTPException(
                status_code=400, 
                detail=f"文件大小超过限制。最大大小: {config['max_file_size'] // (1024*1024)}MB"
            )
        
        # 创建上传目录
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        
        # 保存文件
        file_path = upload_dir / f"{uuid.uuid4()}_{file.filename}"
        content = await file.read()
        
        with open(file_path, "wb") as buffer:
            buffer.write(content)
        
        end_time = time.time()
        success = True
        update_stats(success, end_time - start_time)
        
        return {
            "success": True,
            "file_path": str(file_path),
            "filename": file.filename,
            "file_size": len(content),
            "message": "音频文件上传成功"
        }
        
    except Exception as e:
        end_time = time.time()
        update_stats(success, end_time - start_time)
        logger.error(f"文件上传失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"文件上传失败: {str(e)}")

@app.get("/download/{filename:path}")
async def download_audio(filename: str):
    """下载生成的音频文件"""
    file_path = Path("outputs") / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="文件不存在")
    
    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type="audio/wav"
    )

@app.post("/batch_tts")
async def batch_text_to_speech(request: BatchTTSRequest, background_tasks: BackgroundTasks, tts: IndexTTS = Depends(get_tts_instance)):
    """批量文本转语音"""
    start_time = time.time()
    success = False
    
    try:
        results = []
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        
        for i, tts_request in enumerate(request.requests):
            # 生成输出路径
            if not tts_request.output_path:
                tts_request.output_path = str(output_dir / f"batch_{uuid.uuid4()}_{i}.wav")
            
            # 检查参考音频文件
            if not tts_request.audio_prompt_path or not os.path.exists(tts_request.audio_prompt_path):
                results.append({
                    "index": i,
                    "success": False,
                    "error": "参考音频文件不存在"
                })
                continue
            
            try:
                # 执行TTS推理
                item_start_time = time.time()
                result = tts.infer(
                    audio_prompt=tts_request.audio_prompt_path,
                    text=tts_request.text,
                    output_path=tts_request.output_path,
                    verbose=tts_request.verbose,
                    max_text_tokens_per_sentence=tts_request.max_text_tokens_per_sentence,
                    do_sample=tts_request.do_sample,
                    top_p=tts_request.top_p,
                    top_k=tts_request.top_k,
                    temperature=tts_request.temperature,
                    length_penalty=tts_request.length_penalty,
                    num_beams=tts_request.num_beams,
                    repetition_penalty=tts_request.repetition_penalty,
                    max_mel_tokens=tts_request.max_mel_tokens
                )
                item_end_time = time.time()
                
                results.append({
                    "index": i,
                    "success": True,
                    "output_path": tts_request.output_path,
                    "inference_time": round(item_end_time - item_start_time, 2),
                    "text_length": len(tts_request.text)
                })
                
            except Exception as e:
                results.append({
                    "index": i,
                    "success": False,
                    "error": str(e)
                })
        
        end_time = time.time()
        success = True
        update_stats(success, end_time - start_time)
        
        return {
            "success": True,
            "total_requests": len(request.requests),
            "results": results,
            "message": "批量TTS处理完成"
        }
        
    except Exception as e:
        end_time = time.time()
        update_stats(success, end_time - start_time)
        logger.error(f"批量TTS处理失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"批量TTS处理失败: {str(e)}")

@app.get("/list_audio_files")
async def list_audio_files():
    """列出所有生成的音频文件"""
    try:
        output_dir = Path("outputs")
        if not output_dir.exists():
            return {"files": []}
        
        files = []
        for file_path in output_dir.glob("*.wav"):
            stat = file_path.stat()
            files.append({
                "filename": file_path.name,
                "size": stat.st_size,
                "created_time": stat.st_ctime,
                "created_time_str": datetime.fromtimestamp(stat.st_ctime).strftime("%Y-%m-%d %H:%M:%S"),
                "download_url": f"/download/{file_path.name}"
            })
        
        # 按创建时间排序
        files.sort(key=lambda x: x["created_time"], reverse=True)
        
        return {"files": files}
        
    except Exception as e:
        logger.error(f"获取文件列表失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取文件列表失败: {str(e)}")

@app.delete("/delete_audio/{filename:path}")
async def delete_audio_file(filename: str):
    """删除音频文件"""
    try:
        file_path = Path("outputs") / filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="文件不存在")
        
        file_path.unlink()
        return {"success": True, "message": f"文件 {filename} 删除成功"}
        
    except Exception as e:
        logger.error(f"删除文件失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"删除文件失败: {str(e)}")

@app.delete("/cleanup_old_files")
async def cleanup_old_files():
    """清理旧文件"""
    try:
        if not config["auto_cleanup"]:
            raise HTTPException(status_code=400, detail="自动清理功能未启用")
        
        cutoff_time = time.time() - config["max_file_age"]
        cleaned_files = []
        
        # 清理outputs目录
        output_dir = Path("outputs")
        if output_dir.exists():
            for file_path in output_dir.glob("*.wav"):
                if file_path.stat().st_ctime < cutoff_time:
                    file_path.unlink()
                    cleaned_files.append(str(file_path))
        
        # 清理uploads目录
        upload_dir = Path("uploads")
        if upload_dir.exists():
            for file_path in upload_dir.glob("*"):
                if file_path.stat().st_ctime < cutoff_time:
                    file_path.unlink()
                    cleaned_files.append(str(file_path))
        
        return {
            "success": True,
            "cleaned_files": cleaned_files,
            "message": f"清理了 {len(cleaned_files)} 个旧文件"
        }
        
    except Exception as e:
        logger.error(f"清理旧文件失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"清理旧文件失败: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host=config["host"],
        port=config["port"],
        reload=config["reload"],
        log_level=config["log_level"]
    ) 