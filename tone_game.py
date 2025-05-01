import os
import asyncio
import wave
import pyaudio
from challenge_generator import get_random_sentence_with_emotion, load_dataset
from stt import process_audio_and_match  # Assuming this function is defined elsewhere
from hubert_model import predict_tone

class AudioRecorder:
    """Handles recording audio input and saving it as a .wav file."""
    def __init__(self, filename="recording.wav", channels=1, rate=16000, chunk=1024, record_seconds=12):
        self.filename = filename
        self.channels = channels
        self.rate = rate
        self.chunk = chunk
        self.record_seconds = record_seconds
        self.pyaudio = pyaudio.PyAudio()
        self.stream = None
        self.frames = []
        
    def start_recording(self):
        print("Recording started. Speak now...")

        try:
            # Open stream
            self.stream = self.pyaudio.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk
            )
            self.frames = []

            # Record audio for the specified duration
            for _ in range(0, int(self.rate / self.chunk * self.record_seconds)):
                data = self.stream.read(self.chunk)
                self.frames.append(data)

            print("Recording finished.")

        except Exception as e:
            print(f"Error during recording: {e}")

        finally:
            # Stop and close the stream, and terminate PyAudio
            if self.stream is not None:
                self.stream.stop_stream()
                self.stream.close()
            self.pyaudio.terminate()

        # Save the recorded audio to a file
        wf = wave.open(self.filename, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.pyaudio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(self.rate)
        wf.writeframes(b''.join(self.frames))
        wf.close()

        print(f"Recording saved to {self.filename}")
        return self.filename


async def main():
    dataset_path = 'test.csv'  # Update this path to your dataset
    df = load_dataset(dataset_path)

    # Get a random sentence with its core and random emotions
    sentence, core_emotion, random_emotion = get_random_sentence_with_emotion(df)
    print("Challenge sentence:", sentence)
    print("Core Emotion:", core_emotion)
    print("Challenge Emotion:", random_emotion)

    recorder = AudioRecorder()
    
    while True:
        audio_file_path = recorder.start_recording()
        
        
        result = await process_audio_and_match(audio_file_path, sentence) #Process audio and match against challenge sentence
        
        if result:
            print("Transcription matches the challenge statement!")
            audio_file = 'recording.wav'  # Replace with your actual file path

            # Call the function from the imported module
            emotion_probabilities = predict_tone(audio_file)

            # Print the results
            print("-------------------------------------------------------")
            print("Emotion Probabilities from Imported Module:")
            
            for emotion, probability in emotion_probabilities.items():
                print(f"{emotion}: {probability:.3f}")
                if random_emotion.lower() == emotion.lower():
                    score = int(70 * probability) + 30
                    
            print("-------------------------------------------------------")
            print(f"Score: {score:}/100")
                
            break  # Exit loop if transcription matches
        else:
            print("Transcription did not match. Please try again.")
            break
            # await asyncio.sleep(1)  # Wait for 1 second before retrying

if __name__ == "__main__":
    asyncio.run(main())
