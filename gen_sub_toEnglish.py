# Import the module
from pytube import YouTube
import whisper
import torch
import os
from whisper.utils import get_writer

# Initialize the device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the model 
whisper_model = whisper.load_model("small", device=device)

def download_video(video_URL, destination, final_filename):
  video = YouTube(video_URL)
  progMP4 = video.streams.filter(progressive=True, file_extension='mp4')
  #print(progMP4)
  
  targetMP4 = progMP4.order_by('resolution').desc().first()
  #print(targetMP4)
  
  video_file = targetMP4.download()
  #print(video_file)
  
  _, ext = os.path.splitext(video_file)
  new_file = final_filename + '.mp4' 
  
  # Change the name of the file
  os.rename(video_file, new_file) 
 
  
# Video to Audio
video_URL = 'https://www.youtube.com/watch?v=4nWIc5FRFqM&t=24s'
destination = "."
final_filename = "mininet-wifi3"
download_video(video_URL, destination, final_filename)
print("download is done")

mp4file = final_filename + ".mp4"
result = whisper_model.transcribe(mp4file, task='translate')

# Print the final result
#print(result["text"])

# save SRT
srt_writer = get_writer("srt", destination)
srt_writer(result, final_filename)
