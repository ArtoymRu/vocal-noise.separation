import subprocess

def extract_audio(video_path, output_path):
    '''
    Extract audio from a video file using ffmpeg
    
    :param video_path: Path to the original video file.
    :param output_path: Path to the extracted audio file.
    :return: output_path.
    '''
    try:
        command = ["ffmpeg", "-i", video_path, "-ab", "160k", "-ac", "2", "-ar", "44100", "-vn", output_path]
        subprocess.run(command, check=True)
        return output_path
    except Exception as e:
        return str(e)