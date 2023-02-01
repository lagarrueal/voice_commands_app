import streamlit as st
import numpy as np
import tensorflow as tf
import plotly.graph_objs as go
import plotly.express as px
import librosa
from audio_recorder_streamlit import audio_recorder
import os

def decode_audio(audio_binary):
    # Decode WAV-encoded audio files to `float32` tensors, 
    # normalized to the [-1.0, 1.0] range. Return `float32` audio and a sample rate.
    try :
        audio, _ = tf.audio.decode_wav(contents=audio_binary)
    except :
        return None
    # Since all the data is single channel (mono), drop the `channels`
    # axis from the array.
    return tf.squeeze(audio, axis=-1)

def get_spectrogram(waveform):
    # Zero-padding for an audio waveform with less than 16,000 samples.
    input_len = 16000
    waveform = waveform[:input_len]
    zero_padding = tf.zeros(
        [16000] - tf.shape(waveform),
        dtype=tf.float32)
    # Cast the waveform tensors' dtype to float32.
    waveform = tf.cast(waveform, dtype=tf.float32)
    # Concatenate the waveform with `zero_padding`, which ensures all audio
    # clips are of the same length.
    equal_length = tf.concat([waveform, zero_padding], 0)
    # Convert the waveform to a spectrogram via a STFT.
    spectrogram = tf.signal.stft(
        equal_length, frame_length=255, frame_step=128)
    # Obtain the magnitude of the STFT.
    spectrogram = tf.abs(spectrogram)
    # Add a `channels` dimension, so that the spectrogram can be used
    # as image-like input data with convolution layers (which expect
    # shape (`batch_size`, `height`, `width`, `channels`).
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram

# The code below is for the title and logo for this page.
st.set_page_config(page_title="Commands Recognition App", page_icon="💬")

st.image(
    "assets/voice_recognition.jpg",
    width=160,
)

st.title("`Commands Recognition App` 💬 ")

st.write("")

st.markdown(
    """
    By Axel BOURRAS, Alexandre LAGARRUE, Jules LEFEBVRE, Augustin NESSON
"""
)

st.write("")

option = st.selectbox(
    'Please select an audio file to try the model:',
    os.listdir('data/'))

st.write('You selected:', option)