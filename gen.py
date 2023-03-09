import whisper
import torch
from pathlib import Path
from whisper.utils import get_writer

# GlobalVariable
Model_Type = "small"
Data_File = "dockernet.mp4"
TextFileName = Path(Data_File).stem
file_name = f"{TextFileName}"
output_directory = "."

# check if you have a GPU available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#load Whipser model
model = whisper.load_model(Model_Type, device=DEVICE)
result = model.transcribe(Data_File)

# save TXT
#txt_writer = get_writer("txt", output_directory)
#txt_writer(result, file_name)

# save SRT
srt_writer = get_writer("srt", output_directory)
srt_writer(result, file_name)
