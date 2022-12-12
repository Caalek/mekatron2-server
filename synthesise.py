import sys
import os
sys.path.append("mekatron2")
sys.path.append(os.path.join("mekatron2", 'waveglow/'))
import time
import matplotlib
import matplotlib.pylab as plt
import numpy as np
import torch
import IPython.display as ipd

from pydub import AudioSegment
import scipy.io.wavfile
from scipy.io.wavfile import write

from hparams import create_hparams
from model import Tacotron2
from layers import TacotronSTFT
from audio_processing import griffin_lim
from text import text_to_sequence
from denoiser import Denoiser
from pydub import AudioSegment
import scipy.io.wavfile
from scipy.io.wavfile import write

def generate_voice(text_to_generate, voice):
    
    tacotron2_pretrained_model = os.path.join("models", voice)

    waveglow_pretrained_model = 'models/waveglow'

    thisdict = {}
    for line in reversed((open('merged.dict.txt', "r").read()).splitlines()):
        thisdict[(line.split(" ",1))[0]] = (line.split(" ",1))[1].strip()

    def ARPA(text):
        out = ''
        for word_ in text.split(" "):
            word=word_; end_chars = ''
            while any(elem in word for elem in r"!?,.;") and len(word) > 1:
                if word[-1] == '!': end_chars = '!' + end_chars; word = word[:-1]
                if word[-1] == '?': end_chars = '?' + end_chars; word = word[:-1]
                if word[-1] == ',': end_chars = ',' + end_chars; word = word[:-1]
                if word[-1] == '.': end_chars = '.' + end_chars; word = word[:-1]
                if word[-1] == ';': end_chars = ';' + end_chars; word = word[:-1]
                else: break
            try: word_arpa = thisdict[word.upper()]
            except: word_arpa = ''
            if len(word_arpa)!=0: word = "{" + str(word_arpa) + "}"
            out = (out + " " + word + end_chars).strip()
        if out[-1] != ";": out = out + ";"
        return out

    hparams = create_hparams()

    hparams.sampling_rate = 22050
    hparams.max_decoder_steps = 5000
    hparams.gate_threshold = 0.1

    device = torch.device('cpu')

    model = Tacotron2(hparams)
    model.load_state_dict(torch.load(tacotron2_pretrained_model, map_location=device)['state_dict'])
    _ = model.eval()

    # Załaduj Waveglow
    waveglow = torch.load(waveglow_pretrained_model, map_location=device)['model']
    denoiser = Denoiser(waveglow)

    denoise_strength =  0.02
    equalize = True
    gan = True
    _sigma = 1
    speed_multiplier = 1
    raw_input = False 
    
    for i in text_to_generate.split("\n"):
        if len(i) < 1: 
            continue
        print(i)
        if raw_input:
            if i[-1] != ";": 
                i=i+";" 
        else: 
            i = ARPA(i)
        print(i)
        
        sequence = np.array(text_to_sequence(text_to_generate, ['basic_cleaners']))[None, :]
        sequence = torch.autograd.Variable(
        torch.from_numpy(sequence)).long()
        mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
        with torch.no_grad():
            audio = waveglow.infer(mel_outputs_postnet, sigma=_sigma)
            audio_denoised = denoiser(audio, strength=denoise_strength)[:, 0]
        
        audio = ipd.Audio(audio_denoised.cpu().numpy(), rate=hparams.sampling_rate * speed_multiplier)
        audio = AudioSegment(audio.data[128:], frame_rate=hparams.sampling_rate * speed_multiplier, sample_width=2, channels=1)
        audio.export("test_files/testnt.wav", format="wav", bitrate="32k")

        # coś tu się dzieje
        os.system('ffmpeg -loglevel quiet -y -i "test_files/testnt.wav" -ss 0.0000 -vcodec copy -acodec copy "test_files/test.wav"')
        
        # tutaj EQ
        if equalize:
            os.system('ffmpeg -loglevel quiet -y -i "test_files/test.wav" -ac 2 -af "aresample=44100:resampler=soxr:precision=15, equalizer=f=50:width_type=o:width=0.75:g=3.6, equalizer=f=3000:width_type=o:width=1.0:g=2.0, equalizer=f=10000:width_type=o:width=1.0:g=4.0" "test_EQ.wav"')

        # tutaj gan
        if gan:
            if equalize:
                os.system("python hifi-gan/inference.py --checkpoint_file pretrained")
                os.system('ffmpeg -loglevel quiet -y -i "generated_files/test_generated.wav" -ac 2 -af "aresample=44100:resampler=soxr:precision=15, equalizer=f=50:width_type=o:width=0.75:g=3.6, equalizer=f=3000:width_type=o:width=1.0:g=2.0, equalizer=f=10000:width_type=o:width=1.0:g=4.0" "generated_files/test_EQ.wav"')
        else:
            os.system("python hifi-gan/inference.py --checkpoint_file pretrained")
        return 0



