import pyaudio
import threading
import os
from dotenv import load_dotenv
from deepgram import DeepgramClient
from deepgram.core.events import EventType

class Transcriber:
    def __init__(self):
        print("Initializing Deepgram AI (v7+)...")
        
        load_dotenv()
        api_key = os.getenv("DEEPGRAM_API_KEY")
        if not api_key:
            print("ERROR: DEEPGRAM_API_KEY not found in .env file!")
            self.is_running = False
            return
            
        self.deepgram = DeepgramClient(api_key=api_key)
        self.is_running = False
        
        # Audio configuration
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        self.CHUNK = 1024
        
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.record_thread = None

    def start(self):
        if not hasattr(self, 'deepgram'):
            return
            
        self.is_running = True
        
        # Start PyAudio microphone stream
        self.stream = self.audio.open(format=self.FORMAT,
                                      channels=self.CHANNELS,
                                      rate=self.RATE,
                                      input=True,
                                      frames_per_buffer=self.CHUNK)
        
        # In Deepgram SDK v7+, the live connection is a context manager.
        # We run the context manager inside our recording thread so it doesn't block the main thread.
        self.record_thread = threading.Thread(target=self._record_loop, daemon=True)
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

    def _record_loop(self):
        # Open the Live Transcription (v1) WebSocket connection
        with self.deepgram.listen.v1.connect(
            model="nova-3",
            language="multi",
            smart_format=True,
            interim_results=True,
            encoding="linear16",
            channels=self.CHANNELS,
            sample_rate=self.RATE,
            request_options={"additional_query_parameters": {"language_hint": "zh"}}
        ) as dg_connection:
            
            def on_message(result):
                sentence = result.channel.alternatives[0].transcript
                if len(sentence) == 0:
                    return
                    
                if result.is_final:
                    # Clear the interim line and print the final sentence
                    print(f"\r[Final] {sentence}                                    ")
                else:
                    # Print interim results on the same line using carriage return
                    print(f"\r[Interim] {sentence}", end="", flush=True)

            def on_error(error):
                print(f"Deepgram Error: {error}")

            dg_connection.on(EventType.MESSAGE, on_message)
            dg_connection.on(EventType.ERROR, on_error)
            
            # The v7 SDK requires calling start_listening() to begin processing events,
            # but it is a blocking loop! We must run it in a separate thread so we can send audio.
            listen_thread = threading.Thread(target=dg_connection.start_listening, daemon=True)
            listen_thread.start()
            
            print("Deepgram WebSocket Connected.")
            
            # Blast every tiny 0.06s chunk (1024 bytes) directly to Deepgram!
            while self.is_running:
                try:
                    data = self.stream.read(self.CHUNK, exception_on_overflow=False)
                    dg_connection.send_media(data)
                except Exception as e:
                    print(f"Audio error: {e}")

if __name__ == "__main__":
    import time
    print("Testing Deepgram WebSocket standalone...")
    t = Transcriber()
    t.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        t.stop()
