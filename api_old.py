from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import asyncio
import json
import os
import base64
import time
from typing import Optional, List
from indextts.infer_vllm_v2 import IndexTTS2
import wave


class TTSConfig:
    DEFAULT_EMO_VECTOR: List[float] = [0, 0, 0, 0, 0, 0, 0.45, 0]
    DEFAULT_EMO_ALPHA: float = 0.8
    DEFAULT_DIFFUSION_STEPS: int = 25
    VOICE_PATH: str = 'examples/a_1.mp3'
    CHUNK_DURATION: float = 0.3
    SLEEP_DURATION: float = 0.28
    GPU_MEMORY_UTILIZATION: float = 0.25


class TTSStreamer:
    def __init__(self):
        self.tts = None
        self.last_output_file = None
    
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
    
    async def generate_and_stream(
        self, 
        text: str, 
        websocket: WebSocket,
        emo_vector: Optional[List[float]] = None,
        emo_alpha: Optional[float] = None,
        diffusion_steps: Optional[int] = None,
        request_id: Optional[str] = None
    ):
        if self.tts is None:
            self._init_tts()
            
        if not self.tts:
            await websocket.send_json({"error": "TTS model not initialized"})
            return
            
        try:
            if emo_vector is not None:
                if not isinstance(emo_vector, list):
                    await websocket.send_json({"error": "emo_vector must be a list"})
                    return
                if len(emo_vector) != 8:
                    await websocket.send_json({"error": f"emo_vector must have 8 values, got {len(emo_vector)}"})
                    return
            
            if emo_alpha is not None and not (0.0 <= emo_alpha <= 1.0):
                await websocket.send_json({"error": f"emo_alpha must be between 0.0 and 1.0, got {emo_alpha}"})
                return
            
            if diffusion_steps is not None and not (1 <= diffusion_steps <= 100):
                await websocket.send_json({"error": f"diffusion_steps must be between 1 and 100, got {diffusion_steps}"})
                return
            
            print(f"TTS generation: '{text}', emo_vector: {emo_vector}, emo_alpha: {emo_alpha}, diffusion_steps: {diffusion_steps}")
            
            if not os.path.exists(TTSConfig.VOICE_PATH):
                await websocket.send_json({"error": f"Voice file not found: {TTSConfig.VOICE_PATH}"})
                return
            
            self._cleanup_last_output()
            
            output_path = f"/tmp/tts_output_{int(time.time())}.wav"
            self.last_output_file = output_path
            
            await self.tts.infer(
                spk_audio_prompt=TTSConfig.VOICE_PATH,
                text=text,
                output_path=output_path,
                emo_vector=emo_vector if emo_vector is not None else TTSConfig.DEFAULT_EMO_VECTOR,
                emo_alpha=emo_alpha if emo_alpha is not None else TTSConfig.DEFAULT_EMO_ALPHA,
                diffusion_steps=diffusion_steps if diffusion_steps is not None else TTSConfig.DEFAULT_DIFFUSION_STEPS,
                verbose=True
            )
            
            if not os.path.exists(output_path):
                await websocket.send_json({"error": "TTS failed to generate audio file"})
                return
            
            await self._stream_audio_file(output_path, websocket, request_id)
            
        except Exception as e:
            print(f"TTS Error: {e}")
            import traceback
            traceback.print_exc()
            await websocket.send_json({"error": f"TTS generation failed: {str(e)}"})
    
    def _cleanup_last_output(self):
        if self.last_output_file and os.path.exists(self.last_output_file):
            try:
                os.unlink(self.last_output_file)
            except Exception as e:
                print(f"Cleanup failed: {e}")
    
    async def _stream_audio_file(self, file_path: str, websocket: WebSocket, request_id: str):
        try:
            with wave.open(file_path, 'rb') as wav_file:
                sample_rate = wav_file.getframerate()
                
                await websocket.send_json({
                    "type": "session.created",
                    "session": {
                        "output_audio_format": "pcm16",
                        "sample_rate": sample_rate
                    }
                })
                
                chunk_size = int(sample_rate * TTSConfig.CHUNK_DURATION)
                
                while frames := wav_file.readframes(chunk_size):
                    await websocket.send_json({
                        "type": "response.audio.delta",
                        "response_id": request_id,
                        "request_id": request_id,
                        "item_id": "item_001",
                        "output_index": 0,
                        "content_index": 0,
                        "delta": base64.b64encode(frames).decode('utf-8')
                    })
                    
                    await asyncio.sleep(TTSConfig.SLEEP_DURATION)
                
                await websocket.send_json({
                    "type": "response.audio.done",
                    "response_id": request_id,
                    "request_id": request_id,
                    "item_id": "item_001", 
                    "output_index": 0,
                    "content_index": 0
                })
                
        except Exception as e:
            await websocket.send_json({"type": "error", "error": str(e)})


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
            
            emo_vector = request.get("emo_vector")
            emo_alpha = request.get("emo_alpha")
            diffusion_steps = request.get("diffusion_steps")
            request_id = request.get("request_id")
            
            await tts_streamer.generate_and_stream(text, websocket, emo_vector, emo_alpha, diffusion_steps, request_id)
            
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
    
    <script>
        let ws = null;
        let audioContext = null;
        let sampleRate = 24000;
        let nextTime = 0;
        
        function initWebSocket() {
            ws = new WebSocket(`ws://${window.location.host}/tts`);
            
            ws.onopen = () => {
                document.getElementById('status').textContent = 'Connected';
            };
            
            ws.onmessage = async (event) => {
                const message = JSON.parse(event.data);
                
                if (message.type === 'session.created') {
                    await initAudio(message.session.sample_rate);
                    document.getElementById('status').textContent = 'Audio session started';
                }
                else if (message.type === 'response.audio.delta') {
                    await playAudioDelta(message.delta);
                }
                else if (message.type === 'response.audio.done') {
                    document.getElementById('status').textContent = 'Audio playback complete';
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
            if (audioContext) {
                await audioContext.close();
            }
            
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
            sampleRate = rate;
            
            if (audioContext.state === 'suspended') {
                await audioContext.resume();
            }
            
            nextTime = audioContext.currentTime + 0.5;
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
                
                const source = audioContext.createBufferSource();
                source.buffer = audioBuffer;
                source.connect(audioContext.destination);
                
                const currentTime = audioContext.currentTime;
                if (nextTime <= currentTime) {
                    nextTime = currentTime + 0.05;
                }
                
                source.start(nextTime);
                nextTime += audioBuffer.duration;
                
            } catch (e) {
                console.error('Audio delta error:', e);
            }
        }
        
        function sendText() {
            const text = document.getElementById('textInput').value;
            if (!text) return;
            
            if (!ws || ws.readyState !== WebSocket.OPEN) {
                initWebSocket();
                setTimeout(() => sendText(), 1000);
                return;
            }
            
            const payload = { text };
            
            const emoVectorStr = document.getElementById('emoVector').value.trim();
            if (emoVectorStr) {
                payload.emo_vector = emoVectorStr.split(',').map(v => parseFloat(v.trim()));
            }
            
            const emoAlphaStr = document.getElementById('emoAlpha').value.trim();
            if (emoAlphaStr) {
                const alphaValue = parseFloat(emoAlphaStr);
                if (alphaValue >= 0 && alphaValue <= 1.0) {
                    payload.emo_alpha = alphaValue;
                }
            }
            
            const diffusionStepsStr = document.getElementById('diffusionSteps').value.trim();
            if (diffusionStepsStr) {
                const stepsValue = parseInt(diffusionStepsStr);
                if (stepsValue >= 1 && stepsValue <= 100) {
                    payload.diffusion_steps = stepsValue;
                }
            }
            
            stopAudio();
            ws.send(JSON.stringify(payload));
            document.getElementById('status').textContent = 'Generating audio...';
        }
        
        function stopAudio() {
            if (audioContext) {
                audioContext.close();
                audioContext = null;
            }
            nextTime = 0;
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
