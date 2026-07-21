import pyaudio
import threading
import os
import asyncio
import websockets
import json
import base64
from dotenv import load_dotenv

class Transcriber:
    def __init__(self, language_code="zh", mute_console=False):
        print(f"Initializing ElevenLabs Scribe v2 Realtime (Language: {language_code})...")
        self.language_code = language_code
        self.mute_console = mute_console
        
        load_dotenv()
        self.api_key = os.getenv("ELEVENLABS_API_KEY")
        if not self.api_key:
            print("ERROR: ELEVENLABS_API_KEY not found in .env file!")
            self.is_running = False
            return
            
        self.is_running = False
        self.current_text = ""
        self.previous_text = ""
        self.is_current_final = True
        self.lock = threading.Lock()
        
        # Audio configuration (ElevenLabs accepts linear PCM 16000Hz)
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        self.CHUNK = 2048 # slightly larger chunks for WS
        
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.record_thread = None

    def get_texts(self):
        with self.lock:
            return self.previous_text, self.current_text

    def start(self):
        if not self.api_key:
            return
            
        self.is_running = True
        
        # Start PyAudio microphone stream
        self.stream = self.audio.open(format=self.FORMAT,
                                      channels=self.CHANNELS,
                                      rate=self.RATE,
                                      input=True,
                                      frames_per_buffer=self.CHUNK)
        
        # We run the asyncio event loop inside our recording thread
        self.record_thread = threading.Thread(target=self._run_async_loop, daemon=True)
        self.record_thread.start()
        print("Microphone listening... Speak now!")

    def stop(self):
        print("\nStopping transcriber...")
        self.is_running = False
            
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()
        
        if self.record_thread:
            self.record_thread.join()
        print("Transcriber stopped.")

    def _run_async_loop(self):
        # Create a new event loop for this background thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._async_record_loop())
        loop.close()

    async def _async_record_loop(self):
        url = f"wss://api.elevenlabs.io/v1/speech-to-text/realtime?model_id=scribe_v2_realtime&commit_strategy=vad&language_code={self.language_code}"
        headers = {"xi-api-key": self.api_key}
        
        try:
            async with websockets.connect(url, additional_headers=headers) as ws:
                print("ElevenLabs WebSocket Connected.")
                
                async def sender():
                    while self.is_running:
                        try:
                            # Read audio from mic
                            data = self.stream.read(self.CHUNK, exception_on_overflow=False)
                            # ElevenLabs STT expects a JSON payload with base64 encoded audio
                            payload = {
                                "message_type": "input_audio_chunk",
                                "audio_base_64": base64.b64encode(data).decode("utf-8"),
                                "sample_rate": self.RATE,
                                "commit": False
                            }
                            await ws.send(json.dumps(payload))
                            await asyncio.sleep(0.01) # Yield context
                        except Exception as e:
                            print(f"Audio sender error: {e}")
                            break
                            
                async def receiver():
                    while self.is_running:
                        try:
                            # Use wait_for so we periodically check self.is_running
                            # and don't hang indefinitely on Ctrl+C
                            try:
                                msg_str = await asyncio.wait_for(ws.recv(), timeout=0.5)
                            except asyncio.TimeoutError:
                                continue
                                
                            msg = json.loads(msg_str)
                            
                            msg_type = msg.get("message_type", msg.get("type", ""))
                            
                            # Handle different response types from ElevenLabs
                            if msg_type == "partial_transcript":
                                text = msg.get("text", "")
                                if text:
                                    with self.lock:
                                        if self.is_current_final:
                                            self.previous_text = self.current_text
                                            self.is_current_final = False
                                        self.current_text = text
                                if not self.mute_console:
                                    print(f"\r[Interim] {text}", end="", flush=True)
                            elif msg_type == "committed_transcript" or msg_type == "final_transcript":
                                text = msg.get("text", "")
                                if text:
                                    with self.lock:
                                        if self.is_current_final:
                                            self.previous_text = self.current_text
                                        self.current_text = text
                                        self.is_current_final = True
                                if not self.mute_console:
                                    print(f"\r[Final] {text}                                    ")
                            elif msg_type != "session_started":
                                # Catch-all debug for unexpected messages from ElevenLabs
                                print(f"\n[DEBUG] Unknown message: {msg_str}")
                                is_final = msg.get("is_final", False)
                                text = msg.get("text", "")
                                if text:
                                    with self.lock:
                                        if self.is_current_final and not is_final:
                                            self.previous_text = self.current_text
                                            self.is_current_final = False
                                        elif self.is_current_final and is_final:
                                            self.previous_text = self.current_text
                                        self.current_text = text
                                        if is_final:
                                            self.is_current_final = True
                                if not self.mute_console:
                                    if is_final:
                                        print(f"\r[Final] {text}                                    ")
                                    else:
                                        print(f"\r[Interim] {text}", end="", flush=True)
                                
                        except websockets.exceptions.ConnectionClosed:
                            print("\nWebSocket connection closed by server.")
                            break
                        except Exception as e:
                            print(f"\nReceiver error: {e}")
                            break

                await asyncio.gather(sender(), receiver())
                
        except Exception as e:
            print(f"ElevenLabs WebSocket error: {e}")

if __name__ == "__main__":
    import time
    print("Testing ElevenLabs WebSocket standalone...")
    t = Transcriber()
    t.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        t.stop()
