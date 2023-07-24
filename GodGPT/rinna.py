# -*- coding: utf-8 -*-

import subprocess
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM,Wav2Vec2ForCTC,Wav2Vec2Processor
import socket
import re
import time
import numpy as np
import pyopenjtalk
from scipy.io import wavfile
import pygame
import gc
from io import BytesIO
import librosa
import i2clcda as lcd
import ctranslate2
global talk
talk=True
from faster_whisper import WhisperModel
import re
import streamlit as st
import logging

def jtalk(t,l):
    #lcd.lcd_init()
    #lcd.lcd_string("Answering...",lcd.LCD_LINE_1)
    global talk
    talk=False
    fname="test.wav"
    x,sr=pyopenjtalk.tts(t)
    wavfile.write(fname,sr,x.astype(np.int16))
    pygame.mixer.init(frequency=44100)
    pygame.mixer.music.load(fname)
    pygame.mixer.music.play()
    time.sleep(l/5)
    talk=True
    
def jtalk2(t):
    #lcd.lcd_init()
    #lcd.lcd_string("Answering...",lcd.LCD_LINE_1)
    global talk
    talk=False
    fname="test.wav"
    x,sr=pyopenjtalk.tts(t)
    wavfile.write(fname,sr,x.astype(np.int16))
    pygame.mixer.init(frequency=44100)
    pygame.mixer.music.load(fname)
    pygame.mixer.music.play()
    time.sleep(3)
    talk=True

lcd.lcd_init()
lcd.lcd_string("Please Wait...",lcd.LCD_LINE_1)
lcd.lcd_string("Model Setting",lcd.LCD_LINE_2)
jtalk2("モデルを起動します")

import gc

tokenizer=AutoTokenizer.from_pretrained("rinna/japanese-gpt-neox-small",use_fast=False)
gc.collect()
model=AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt-neox-small")
#generator=ctranslate2.Generator("rinna-ppo-short")
gc.collect()
model=WhisperModel("large-v2")
#processor=Wav2Vec2Processor.from_pretrained("NTQAI/wav2vec2-large-japanese")
#wavmodel=Wav2Vec2ForCTC.from_pretrained("NTQAI/wav2vec2-large-japanese")

lcd.lcd_init()
lcd.lcd_string("Wait a minute",lcd.LCD_LINE_1)
jtalk2("起動成功、ようこそゴッドGPTへ")

def generate(prompt):
    global talk
    talk=False
    prompt="Q:あなたは誰ですか？ \nA:私は神だ、万物の神である。 \nQ:なるほど。恐れ多いのですが、対話しませんか？ \nA:いいだろう、何でも答えてやろう。 "+"\nQ:"+prompt+" \nA:"
    
    token_ids=tokenizer.encode(prompt,add_special_tokens=False)
    tokens=tokenizer.convert_ids_to_tokens(token_ids)
    results=generator.generate_batch([tokens],max_length=64,sampling_topk=30,sampling_temperature=0.7)

    output=tokenizer.decode(results[-1].sequences_ids[0])
    output=output.split("A:")
    
    if "Q:" in output[-1]:
        output=output[-1].split("Q:")
        return output[-2]
    else:
        return output[-1]

import os
import speech_recognition as sr
from datetime import datetime


class SpeechRecognizer:
    """マイクで受け取った音声を認識してファイル出力するクラス
    """
    def __init__(self):
        os.makedirs("./out", exist_ok=True)
        self.path = f"./out/asr.txt"
        global rec
        rec=sr.Recognizer()
        rec.energy_threshold=2000
        rec.dynamic_energy_threshold=False
        self.rec = rec
        self.mic = sr.Microphone()
        self.speech = []
        return

    def grab_audio(self) -> sr.AudioData:
        """マイクで音声を受け取る関数

        Returns:
            speech_recognition.AudioData: 音声認識エンジンで受け取った音声データ
        """
        print("何か話してください...")
        with self.mic as source:
            talk=False
            self.rec.adjust_for_ambient_noise(source)
            audio = self.rec.listen(source)
        with open("userwav.wav","wb")as f:
            f.write(audio.get_wav_data())
            
            #audio_bytes=audio.get_raw_data()
            #speech_array=np.frombuffer(audio_bytes,dtype=np.int64)
            #speech_stream=BytesIO()
            #wavfile.write("userwav.wav",44100,speech_array)
    def recognize_audio(self):
        #jtalk("認識...")
        lcd.lcd_init()
        lcd.lcd_string("Recognize...",lcd.LCD_LINE_1)
        segments,info=model.transcribe("userwav.wav",beam_size=5,language="ja")
        speech=""
        for segment in segments:
            speech+=segment.text
        
        
        return speech

    def run(self):
        """マイクで受け取った音声を認識してテキストファイルに出力
        """
        while True:
            global talk
            if talk==True:
                talk=False
                self.grab_audio()
                #print(audio.type)
                speech = self.recognize_audio()
                #speech="テスト"
                if speech == "終了":
                    jtalk("音声認識終了")
                    break
                elif speech=="認識できませんでした":
                    print("認識できませんでした。再び話しかけてください。")
                    talk=True
                
                elif speech=="音声認識のリクエストが失敗しました":
                    jtalk("おそらくネットにつながっていません。つなげてから話しかけてください")
                    break
                else:
                    #self.speech.append(speech)
                    print(speech)
                    st.title(output)
                    output=generate(speech)
                    jtalk(output,len(output))
                    #talk=True
                    print(output)
                    st.title(output)
                    #jtalk("私からは以上です。")
                lcd.lcd_init()
                lcd.lcd_string("Ready",lcd.LCD_LINE_1)

    

if __name__ == "__main__":
    talk=True
    sp = SpeechRecognizer()
    lcd.lcd_init()
    lcd.lcd_string("Ready",lcd.LCD_LINE_1)
    #jtalk("なにか話しかけてみてください。「終了」と話すと音声認識を終了します")
    sp.run()
