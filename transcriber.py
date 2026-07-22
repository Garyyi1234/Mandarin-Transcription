import pyaudio
import threading
import os
import asyncio
import websockets
import json
import base64
import time
from dotenv import load_dotenv
from deep_translator import GoogleTranslator

class Transcriber:
    def __init__(self, language_code="zh", mute_console=False):
        print(f"Initializing ElevenLabs Scribe v2 Realtime (Language: {language_code})...")
        self.language_code = language_code
        self.mute_console = mute_console
        
        load_dotenv()
        self.api_key = os.getenv("ELEVENLABS_API_KEY")
        
        self.translator = GoogleTranslator(source='en', target='zh-CN')
        self.translated_previous = ""
        self.translated_current = ""
        self.last_translated_raw_prev = ""
        self.last_translated_raw_curr = ""
        self._translate_thread = None
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
            prev = self.translated_previous if (self.previous_text and self.previous_text == self.last_translated_raw_prev) else self.previous_text
            curr = self.translated_current if (self.current_text and self.current_text == self.last_translated_raw_curr) else self.current_text
            return prev, curr

    def _translation_worker(self):
        while self.is_running:
            with self.lock:
                raw_curr = self.current_text
                raw_prev = self.previous_text
                is_final = self.is_current_final
                
            # Translate previous if it hasn't been translated
            if raw_prev and raw_prev != self.last_translated_raw_prev:
                # If we just translated this as the current_text, copy it over to save an API call
                if raw_prev == self.last_translated_raw_curr:
                    with self.lock:
                        self.translated_previous = self.translated_current
                        self.last_translated_raw_prev = raw_prev
                else:
                    try:
                        trans = self.translator.translate(raw_prev)
                        with self.lock:
                            self.translated_previous = trans
                            self.last_translated_raw_prev = raw_prev
                    except Exception as e:
                        pass
                        
            # Translate current ONLY if it is final
            if is_final and raw_curr and raw_curr != self.last_translated_raw_curr:
                try:
                    trans = self.translator.translate(raw_curr)
                    with self.lock:
                        self.translated_current = trans
                        self.last_translated_raw_curr = raw_curr
                except Exception as e:
                    pass
            
            time.sleep(0.1)

    def start(self):
        if not self.api_key:
            return
            
        self.is_running = True
        
        self._translate_thread = threading.Thread(target=self._translation_worker, daemon=True)
        self._translate_thread.start()
        
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
        if self._translate_thread:
            self._translate_thread.join()
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
