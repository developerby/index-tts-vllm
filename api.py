from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import asyncio
import json
import os
import base64
import gc
from typing import Optional, List
from dataclasses import dataclass, field
from indextts.infer_vllm_v2 import IndexTTS2
import wave
from uuid import uuid4
from enum import Enum
from pathlib import Path


class Voice(str, Enum):
    HONEY = "honey"
    SUNSHINE = "sunshine"
    CHAN = "chan"
    ENDRA = "endra"
    CHLOE = "chloe"


class TTSConfig:
    DEFAULT_EMO_VECTOR: List[float] = [0, 0, 0, 0, 0, 0, 0, 0]
    DEFAULT_EMO_ALPHA: float = 0.0
    DEFAULT_DIFFUSION_STEPS: int = 25
    DEFAULT_VOICE: Voice = Voice.HONEY
    CHUNK_DURATION: float = 0.3
    SLEEP_DURATION: float = 0.01
    GPU_MEMORY_UTILIZATION: float = 0.10
    
    VOICE_MAPPING: dict[Voice, str] = {
        Voice.HONEY: "voice_2.mp3",
        Voice.SUNSHINE: "voice_british_f_001.wav",
        Voice.CHAN: "asian_eng_m_001.wav",
        Voice.ENDRA: "eng_f_bassy_001.wav",
        Voice.CHLOE: "eng_f_bassy_002.wav",
    }
    
    @classmethod
    def get_voice_path(cls, voice: Optional[str] = None) -> str:
        if not voice:
            voice_enum = cls.DEFAULT_VOICE
        else:
            try:
                voice_enum = Voice(voice.lower())
            except ValueError:
                voice_enum = cls.DEFAULT_VOICE
        
        filename = cls.VOICE_MAPPING[voice_enum]
        return str(Path("examples") / filename)


@dataclass
class TTSRequest:
    request_id: str
    sentence_id: str
    text: str
    voice: Optional[str]
    emo_vector: Optional[List[float]]
    emo_alpha: Optional[float]
    diffusion_steps: Optional[int]
    websocket: WebSocket
    output_path: Optional[str] = None
    order_index: int = 0
    is_last_in_session: bool = False
    
    def cleanup(self):
        if self.output_path and os.path.exists(self.output_path):
            try:
                os.unlink(self.output_path)
                print(f"Deleted: {self.output_path}")
            except Exception as e:
                print(f"Cleanup error: {e}")


@dataclass
class QueueStatus:
    generation_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    streaming_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    order_counter: int = 0
    is_processing: bool = False
    is_streaming: bool = False


class TTSStreamer:
    def __init__(self):
        self.tts = None
        self.queue_status = QueueStatus()
    
    def _init_tts(self):
        if self.tts is not None:
            return
            
        try:
            model_dir = os.path.abspath("checkpoints/IndexTTS-2-vLLM")
            if not os.path.exists(os.path.join(model_dir, "config.yaml")):
                raise FileNotFoundError(f"config.yaml not found in {model_dir}")
            
            self.tts = IndexTTS2(
                model_dir=model_dir,
                is_fp16=True,
                use_cuda_kernel=False,
                gpu_memory_utilization=TTSConfig.GPU_MEMORY_UTILIZATION
            )
        except Exception as e:
            print(f"TTS import error: {e}")
            raise
    
    async def enqueue_request(
        self, 
        text: str, 
        websocket: WebSocket,
        voice: Optional[str] = None,
        emo_vector: Optional[List[float]] = None,
        emo_alpha: Optional[float] = None,
        diffusion_steps: Optional[int] = None,
        request_id: Optional[str] = None,
        sentence_id: Optional[str] = None,
        is_last_in_session: bool = False
    ):
        if self.tts is None:
            self._init_tts()
        
        if not request_id:
            request_id = str(uuid4())
        
        if not sentence_id:
            sentence_id = str(uuid4())
        
        order_index = self.queue_status.order_counter
        self.queue_status.order_counter += 1
        
        request = TTSRequest(
            request_id=request_id,
            sentence_id=sentence_id,
            text=text,
            voice=voice,
            emo_vector=emo_vector,
            emo_alpha=emo_alpha,
            diffusion_steps=diffusion_steps,
            websocket=websocket,
            order_index=order_index,
            is_last_in_session=is_last_in_session
        )
        
        await self.queue_status.generation_queue.put(request)
        await websocket.send_json({
            "type": "request.queued",
            "request_id": request_id,
            "sentence_id": sentence_id,
            "queue_position": order_index,
            "queue_size": self.queue_status.generation_queue.qsize()
        })
        
        if not self.queue_status.is_processing:
            asyncio.create_task(self._process_generation_queue())
        
        if not self.queue_status.is_streaming:
            asyncio.create_task(self._process_streaming_queue())
    
    async def _process_generation_queue(self):
        if self.queue_status.is_processing:
            return
        
        self.queue_status.is_processing = True
        
        try:
            while not self.queue_status.generation_queue.empty():
                request = await self.queue_status.generation_queue.get()
                
                try:
                    await request.websocket.send_json({
                        "type": "generation.started",
                        "request_id": request.request_id,
                        "sentence_id": request.sentence_id
                    })
                    
                    output_path = f"/tmp/tts_output_{request.sentence_id.replace('_', '-').replace('.', '-')}.wav"
                    
                    print(f"Generating audio to: {output_path}")
                    
                    await self.tts.infer(
                        spk_audio_prompt=TTSConfig.get_voice_path(request.voice),
                        text=request.text,
                        output_path=output_path,
                        emo_vector=request.emo_vector if request.emo_vector is not None else TTSConfig.DEFAULT_EMO_VECTOR,
                        emo_alpha=request.emo_alpha if request.emo_alpha is not None else TTSConfig.DEFAULT_EMO_ALPHA,
                        diffusion_steps=request.diffusion_steps if request.diffusion_steps is not None else TTSConfig.DEFAULT_DIFFUSION_STEPS,
                        verbose=True
                    )
                    
                    print(f"File exists after generation: {os.path.exists(output_path)}")
                    
                    if os.path.exists(output_path):
                        request.output_path = output_path
                        await self.queue_status.streaming_queue.put(request)
                        
                        await request.websocket.send_json({
                            "type": "generation.completed",
                            "request_id": request.request_id,
                            "sentence_id": request.sentence_id
                        })
                    else:
                        await request.websocket.send_json({
                            "type": "error",
                            "request_id": request.request_id,
                            "sentence_id": request.sentence_id,
                            "error": "Generation failed"
                        })
                        
                except Exception as e:
                    print(f"Generation error for {request.sentence_id}: {e}")
                    import traceback
                    traceback.print_exc()
                    await request.websocket.send_json({
                        "type": "error",
                        "request_id": request.request_id,
                        "sentence_id": request.sentence_id,
                        "error": str(e)
                    })
                    request.cleanup()
                finally:
                    self.queue_status.generation_queue.task_done()
                
        finally:
            self.queue_status.is_processing = False
            gc.collect()
    
    async def _process_streaming_queue(self):
        if self.queue_status.is_streaming:
            return
        
        self.queue_status.is_streaming = True
        session_initialized = False
        
        try:
            pending_requests = []
            
            while True:
                if not self.queue_status.streaming_queue.empty():
                    request = await self.queue_status.streaming_queue.get()
                    pending_requests.append(request)
                    self.queue_status.streaming_queue.task_done()
                
                elif not pending_requests:
                    if self.queue_status.generation_queue.empty() and not self.queue_status.is_processing:
                        break
                    await asyncio.sleep(0.1)
                    continue
                
                pending_requests.sort(key=lambda r: r.order_index)
                
                if pending_requests:
                    request = pending_requests.pop(0)
                    
                    try:
                        if not session_initialized:
                            with wave.open(request.output_path, 'rb') as wav_file:
                                sample_rate = wav_file.getframerate()
                            
                            await request.websocket.send_json({
                                "type": "session.created",
                                "session": {
                                    "output_audio_format": "pcm16",
                                    "sample_rate": sample_rate
                                }
                            })
                            session_initialized = True
                        
                        print(f"Streaming from: {request.output_path}, exists: {os.path.exists(request.output_path)}")
                        
                        await self._stream_audio_file(
                            request.output_path,
                            request.websocket,
                            request.request_id,
                            request.sentence_id,
                            request.is_last_in_session
                        )
                            
                    except Exception as e:
                        print(f"Streaming error for {request.sentence_id}: {e}")
                        import traceback
                        traceback.print_exc()
                        await request.websocket.send_json({
                            "type": "error",
                            "request_id": request.request_id,
                            "sentence_id": request.sentence_id,
                            "error": f"Streaming failed: {str(e)}"
                        })
                    finally:
                        request.cleanup()
                
        finally:
            self.queue_status.is_streaming = False
            gc.collect()
    
    async def _stream_audio_file(self, file_path: str, websocket: WebSocket, request_id: str, sentence_id: str, is_last: bool):
        try:
            with wave.open(file_path, 'rb') as wav_file:
                sample_rate = wav_file.getframerate()
                
                chunk_size = int(sample_rate * TTSConfig.CHUNK_DURATION)
                
                while frames := wav_file.readframes(chunk_size):
                    await websocket.send_json({
                        "type": "response.audio.delta",
                        "request_id": request_id,
                        "sentence_id": sentence_id,
                        "delta": base64.b64encode(frames).decode('utf-8')
                    })
                    
                    await asyncio.sleep(TTSConfig.SLEEP_DURATION)
                
                if is_last:
                    await websocket.send_json({
                        "type": "response.audio.done",
                        "request_id": request_id
                    })
                
        except Exception as e:
            await websocket.send_json({
                "type": "error",
                "request_id": request_id,
                "sentence_id": sentence_id,
                "error": str(e)
            })


app = FastAPI(title="TTS vLLM Streaming API")
tts_streamer = None


@app.on_event("startup")
async def startup_event():
    global tts_streamer
    tts_streamer = TTSStreamer()
    print("Initializing TTS model...")
    tts_streamer._init_tts()
    print("TTS model initialized successfully")


@app.websocket("/tts")
async def websocket_tts_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    try:
        while True:
            data = await websocket.receive_text()
            request = json.loads(data)
            
            text = request.get("text")
            if not text:
                await websocket.send_json({"error": "Text parameter is required"})
                continue
            
            voice = request.get("voice")
            emo_vector = request.get("emo_vector")
            emo_alpha = request.get("emo_alpha")
            diffusion_steps = request.get("diffusion_steps")
            request_id = request.get("request_id")
            sentence_id = request.get("sentence_id")
            is_last_in_session = request.get("is_last_in_session", False)
            
            await tts_streamer.enqueue_request(
                text, websocket, voice, emo_vector, emo_alpha, diffusion_steps, 
                request_id, sentence_id, is_last_in_session
            )
            
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        await websocket.send_json({"error": f"Connection error: {str(e)}"})


@app.get("/")
async def get_test_page():
    html_content = """<!DOCTYPE html>
<html>
<head>
    <title>TTS vLLM Streaming Test</title>
</head>
<body>
    <h1>TTS vLLM Streaming Test</h1>
    <div style="margin: 20px;">
        <label>Text:</label><br>
        <input type="text" id="textInput" placeholder="Enter text" style="width: 400px;"><br><br>
        
        <label>Voice:</label><br>
        <select id="voiceSelect" style="width: 400px;">
            <option value="honey">Honey (voice_2)</option>
            <option value="sunshine">Sunshine (voice_british_f_001)</option>
            <option value="chan">Chan (asian_eng_m_001)</option>
            <option value="endra">Endra (eng_f_bassy_001)</option>
            <option value="chloe">Chloe (eng_f_bassy_001)</option>
        </select><br><br>
        
        <label>Emotion Vector (8 values):</label><br>
        <input type="text" id="emoVector" placeholder="e.g., 0,0,0,0,0,0,0.45,0" style="width: 400px;"><br><br>
        
        <label>Emotion Alpha (0.0-1.0):</label><br>
        <input type="number" id="emoAlpha" step="0.01" min="0" max="1.0" value="0.8" style="width: 400px;"><br><br>
        
        <label>Diffusion Steps (1-100):</label><br>
        <input type="number" id="diffusionSteps" step="1" min="1" max="100" value="25" style="width: 400px;"><br><br>
        
        <button onclick="sendText()">Generate & Stream</button>
        <button onclick="stopAudio()">Stop</button>
    </div>
    <br>
    <div id="status">Ready</div>
    <div id="queue">Queue: 0</div>
    
    <script>
        let ws = null;
        let audioContext = null;
        let sampleRate = 24000;
        let nextTime = 0;
        let audioQueue = [];
        let isPlaying = false;
        
        function initWebSocket() {
            ws = new WebSocket(`ws://${window.location.host}/tts`);
            
            ws.onopen = () => {
                document.getElementById('status').textContent = 'Connected';
            };
            
            ws.onmessage = async (event) => {
                const message = JSON.parse(event.data);
                
                if (message.type === 'request.queued') {
                    document.getElementById('status').textContent = `Queued (position: ${message.queue_position})`;
                    document.getElementById('queue').textContent = `Queue: ${message.queue_size}`;
                }
                else if (message.type === 'generation.started') {
                    document.getElementById('status').textContent = 'Generating audio...';
                }
                else if (message.type === 'generation.completed') {
                    document.getElementById('status').textContent = 'Waiting to stream...';
                }
                else if (message.type === 'session.created') {
                    stopAudio();
                    await initAudio(message.session.sample_rate);
                    document.getElementById('status').textContent = 'Streaming...';
                }
                else if (message.type === 'response.audio.delta') {
                    playAudioDelta(message.delta);
                }
                else if (message.type === 'response.audio.done') {
                    if (audioQueue.length > 0) {
                        playQueue();
                    }
                    document.getElementById('status').textContent = 'Complete';
                }
                else if (message.error) {
                    document.getElementById('status').textContent = 'Error: ' + message.error;
                }
            };
            
            ws.onerror = () => {
                document.getElementById('status').textContent = 'Connection error';
            };
        }
        
        async function initAudio(rate) {
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
            sampleRate = rate;
            
            if (audioContext.state === 'suspended') {
                await audioContext.resume();
            }
            
            nextTime = audioContext.currentTime + 0.3;
            audioQueue = [];
            isPlaying = false;
        }
        
        async function playAudioDelta(base64Audio) {
            if (!audioContext || !base64Audio) return;
            
            try {
                const binaryString = atob(base64Audio);
                const bytes = new Uint8Array(binaryString.length);
                for (let i = 0; i < binaryString.length; i++) {
                    bytes[i] = binaryString.charCodeAt(i);
                }
                
                const pcm16 = new Int16Array(bytes.buffer);
                const float32 = new Float32Array(pcm16.length);
                for (let i = 0; i < pcm16.length; i++) {
                    float32[i] = pcm16[i] / 32768.0;
                }
                
                const audioBuffer = audioContext.createBuffer(1, float32.length, sampleRate);
                audioBuffer.getChannelData(0).set(float32);
                
                audioQueue.push(audioBuffer);
                
                if (!isPlaying && audioQueue.length >= 3) {
                    isPlaying = true;
                    playQueue();
                }
                
            } catch (e) {
                console.error('Audio error:', e);
            }
        }
        
        function playQueue() {
            if (!audioContext || audioQueue.length === 0) {
                isPlaying = false;
                return;
            }
            
            const currentTime = audioContext.currentTime;
            
            if (nextTime < currentTime) {
                nextTime = currentTime;
            }
            
            while (audioQueue.length > 0) {
                const audioBuffer = audioQueue.shift();
                const source = audioContext.createBufferSource();
                source.buffer = audioBuffer;
                source.connect(audioContext.destination);
                
                source.start(nextTime);
                nextTime += audioBuffer.duration;
            }
            
            isPlaying = false;
        }
        
        function sendText() {
            const text = document.getElementById('textInput').value;
            if (!text) return;
            
            if (!ws || ws.readyState !== WebSocket.OPEN) {
                initWebSocket();
                setTimeout(() => sendText(), 1000);
                return;
            }
            
            const requestId = Date.now().toString();
            
            const sentences = text
                .split('.')
                .map(s => s.trim())
                .filter(s => s.length > 0);
            
            if (sentences.length === 0) return;
            
            const voice = document.getElementById('voiceSelect').value;
            
            const emoVectorStr = document.getElementById('emoVector').value.trim();
            const emoVector = emoVectorStr 
                ? emoVectorStr.split(',').map(v => parseFloat(v.trim()))
                : null;
            
            const emoAlphaStr = document.getElementById('emoAlpha').value.trim();
            const emoAlpha = emoAlphaStr 
                ? parseFloat(emoAlphaStr)
                : null;
            
            const diffusionStepsStr = document.getElementById('diffusionSteps').value.trim();
            const diffusionSteps = diffusionStepsStr 
                ? parseInt(diffusionStepsStr)
                : null;
            
            sentences.forEach((sentence, index) => {
                const payload = {
                    text: sentence,
                    voice: voice,
                    request_id: requestId,
                    sentence_id: `${requestId}_${index}`,
                    is_last_in_session: index === sentences.length - 1
                };
                
                if (emoVector) payload.emo_vector = emoVector;
                if (emoAlpha !== null && emoAlpha >= 0 && emoAlpha <= 1.0) {
                    payload.emo_alpha = emoAlpha;
                }
                if (diffusionSteps !== null && diffusionSteps >= 1 && diffusionSteps <= 100) {
                    payload.diffusion_steps = diffusionSteps;
                }
                
                ws.send(JSON.stringify(payload));
            });
            
            document.getElementById('status').textContent = `Sending ${sentences.length} sentences...`;
        }
        
        function stopAudio() {
            if (audioContext) {
                audioContext.close();
                audioContext = null;
            }
            nextTime = 0;
            audioQueue = [];
            isPlaying = false;
            document.getElementById('status').textContent = 'Stopped';
        }
        
        initWebSocket();
    </script>
</body>
</html>"""
    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    import uvicorn
    import sys
    
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 9000
    print(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
