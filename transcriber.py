import pyaudio
import numpy as np
import threading
import queue
from faster_whisper import WhisperModel

class Transcriber:
    def __init__(self, model_size="small", device="cpu", compute_type="int8"):
        print(f"Initializing Whisper model ({model_size}) on {device}...")
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        print("Model loaded.")
        
        # Audio configuration
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        self.CHUNK = 1024
        
        self.audio = pyaudio.PyAudio()
        self.stream = None
        
        self.audio_queue = queue.Queue()
        self.is_running = False
        self.record_thread = None
        self.transcribe_thread = None

        # The initial prompt to help with code-switching
        self.initial_prompt = "You are a bilingual transcriber. This audio contains both Mandarin and English, mainly mandarin with certain nouns in English"

    def start(self):
        self.is_running = True
        self.stream = self.audio.open(format=self.FORMAT,
                                      channels=self.CHANNELS,
                                      rate=self.RATE,
                                      input=True,
                                      frames_per_buffer=self.CHUNK)
        
        self.record_thread = threading.Thread(target=self._record_loop, daemon=True)
        self.transcribe_thread = threading.Thread(target=self._transcribe_loop, daemon=True)
        
        self.record_thread.start()
        self.transcribe_thread.start()
        print("Transcription started.")

    def stop(self):
        print("Stopping transcriber...")
        self.is_running = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()
        
        if self.record_thread:
            self.record_thread.join()
        if self.transcribe_thread:
            # Put a dummy item to unblock the queue if it's empty
            self.audio_queue.put(None)
            self.transcribe_thread.join()
        print("Transcriber stopped.")

    def _record_loop(self):
        # We record in chunks and send them to the transcription thread.
        # Let's accumulate 1.5 seconds of audio at a time for faster responsiveness.
        buffer_frames = []
        frames_per_buffer = int(self.RATE / self.CHUNK * 1.5) # 1.5 seconds
        
        while self.is_running:
            try:
                data = self.stream.read(self.CHUNK, exception_on_overflow=False)
                buffer_frames.append(data)
                
                if len(buffer_frames) >= frames_per_buffer:
                    # Combine chunks
                    audio_data = b''.join(buffer_frames)
                    buffer_frames = []
                    
                    # Convert to numpy array of floats between -1 and 1 (required by faster-whisper)
                    audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                    self.audio_queue.put(audio_np)
            except Exception as e:
                print(f"Audio recording error: {e}")

    def _transcribe_loop(self):
        while self.is_running:
            audio_np = self.audio_queue.get()
            if audio_np is None:
                break # Stop signal
                
            # Perform VAD (Voice Activity Detection) - simple threshold
            # If the audio is completely silent, skip transcription to save CPU
            if np.max(np.abs(audio_np)) < 0.01:
                continue

            try:
                # Transcribe the audio chunk
                segments, info = self.model.transcribe(
                    audio_np, 
                    beam_size=5, 
                    initial_prompt=self.initial_prompt,
                    condition_on_previous_text=False
                )
                
                for segment in segments:
                    text = segment.text.strip()
                    if text:
                        print(f"[{info.language}] {text}")
            except Exception as e:
                print(f"Transcription error: {e}")

if __name__ == "__main__":
    import time
    print("Testing transcriber standalone...")
    t = Transcriber()
    t.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        t.stop()
