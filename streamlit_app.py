import os

import streamlit               as st
import numpy                   as np
import pandas                  as pd
import tensorflow              as tf
import plotly.graph_objs       as go
import plotly.express          as px

from audio_recorder_streamlit import audio_recorder



def decode_audio(audio_binary):
    # Decode WAV-encoded audio files to `float32` tensors, 
    # normalized to the [-1.0, 1.0] range. Return `float32` audio and a sample rate.
    try :
        audio, _ = tf.audio.decode_wav(contents=audio_binary, desired_channels=1)
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

MODEL_PATH = 'models/model_cnn.h5'
DATA_PATH = 'data/'
IMAGE_PATH = 'assets/classes/'

MODEL = tf.keras.models.load_model(MODEL_PATH)

TARGETS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "unknown"]

# The code below is for the title and logo for this page.
st.set_page_config(page_title="Commands Recognition App", page_icon="💬")

st.image(
    "assets/voice_recognition.jpg"
)

st.title("`Commands Recognition App` 💬 ")

st.write("")

st.markdown(
    """
    By Axel BOURRAS, Alexandre LAGARRUE, Jules LEFEBVRE, Augustin NESSON
"""
)


options = os.listdir(DATA_PATH)
options.insert(0, "Select a file")

container_test = st.container()
container_upload = st.container()
container_record = st.container()

with container_test:
    st.title("Test the model with our audio files")
    option = st.selectbox('Please select an audio file to try the model:', options)
    if option != "Select a file":
        st.write('You selected:', option)
        st.write("Listen to the audio file")
        st.audio("data/" + option, format='audio/wav')
        button = st.button('Try the model')
        if not button:
            st.markdown('##')
            st.markdown('##')
        if button:

            audio = decode_audio(tf.io.read_file("data/" + option))
            spectrogram = get_spectrogram(audio).numpy()
            audio = audio.numpy()
            if len(spectrogram.shape) > 2:
                assert len(spectrogram.shape) == 3
                spectrogram = np.squeeze(spectrogram, axis=-1)
            log_spec = np.log(spectrogram.T + np.finfo(float).eps)
            height = log_spec.shape[0]
            width = log_spec.shape[1]
            X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
            Y = [i for i in range(height)]
            
            df_wf = pd.DataFrame(columns=["time","Amplitude"])
            df_wf["time"] = np.arange(0, len(audio))
            df_wf["Amplitude"] = audio
            fig = px.line(df_wf, x="time", y="Amplitude", title="Audio waveform", width=800, height=400)
            st.plotly_chart(fig)
            
            # Create a trace for the spectrogram
            trace = go.Heatmap( z=log_spec, x=X, y=Y, colorscale='Viridis', showscale=False)
            data=[trace]
            layout = go.Layout( title="Spectrogram", width=800, height=400) 
            fig = go.Figure(data=data, layout=layout)
            st.plotly_chart(fig)
            
            tensor = tf.convert_to_tensor(spectrogram)
            tensor = tf.expand_dims(tensor, 0)
            tensor = np.array(tensor.numpy())
            
            prediction = MODEL.predict(tensor)
            
            # Bar chart of the prediction for each class
            fig = go.Figure(data=[go.Bar(x=TARGETS, y=tf.nn.softmax(prediction[0]))])
            fig.update_layout(title_text='Prediction for each class', width=800, height=400)
            st.plotly_chart(fig)
            st.markdown('##')
            
            best_pred = TARGETS[np.argmax(prediction[0])]
            # st.write("The model predicts: " + best_pred)
            # display the image of the predicted class
            # the images are stored in assets/classes/
            st.image(IMAGE_PATH + best_pred + ".png", caption=f"The model predicts: {best_pred}", width=400, use_column_width=False)
            st.markdown('##')
            st.markdown('##')
            
    else:
        st.markdown('##')
        st.markdown('##')

with container_upload:
    st.title("Upload your own audio file")
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        st.write("Listen to the audio file")
        st.audio(uploaded_file, format='audio/wav')
        
        waveform = decode_audio(uploaded_file.read())
        spectrogram = get_spectrogram(waveform).numpy()
        waveform = waveform.numpy()
        if len(spectrogram.shape) > 2:
            assert len(spectrogram.shape) == 3
            spectrogram = np.squeeze(spectrogram, axis=-1)
        log_spec = np.log(spectrogram.T + np.finfo(float).eps)
        height = log_spec.shape[0]
        width = log_spec.shape[1]
        X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
        Y = [i for i in range(height)]
        
        df_wf = pd.DataFrame(columns=["time","Amplitude"])
        df_wf["time"] = np.arange(0, len(waveform))
        df_wf["Amplitude"] = waveform
        fig = px.line(df_wf, x="time", y="Amplitude", title="Audio waveform", width=800, height=400)
        st.plotly_chart(fig)
        
        # Create a trace for the spectrogram
        trace = go.Heatmap( z=log_spec, x=X, y=Y, colorscale='Viridis', showscale=False)
        data=[trace]
        layout = go.Layout( title="Spectrogram", width=800, height=400) 
        fig = go.Figure(data=data, layout=layout)
        st.plotly_chart(fig)
        
        tensor = tf.convert_to_tensor(spectrogram)
        tensor = tf.expand_dims(tensor, 0)
        tensor = np.array(tensor.numpy())
        
        prediction = MODEL.predict(tensor)
        
        # Bar chart of the prediction for each class
        fig = go.Figure(data=[go.Bar(x=TARGETS, y=tf.nn.softmax(prediction[0]))])
        fig.update_layout(title_text='Prediction for each class', width=800, height=400)
        st.plotly_chart(fig)
        st.markdown('##')
        
        best_pred = TARGETS[np.argmax(prediction[0])]
        # st.write("The model predicts: " + best_pred)
        # display the image of the predicted class
        # the images are stored in assets/classes/
        st.image(IMAGE_PATH + best_pred + ".png", caption=f"The model predicts: {best_pred}", width=400, use_column_width=False)
        st.markdown('##')
        st.markdown('##')
            
    else:
        st.markdown('##')
        st.markdown('##')


with container_record:
    st.title("Record your own audio")
    audio_bytes = audio_recorder(sample_rate=16000, text="Click to start the record, then click again to stop")
    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")
        waveform = decode_audio(audio_bytes)
        spectrogram = get_spectrogram(waveform).numpy()
        waveform = waveform.numpy()
        if len(spectrogram.shape) > 2:
            assert len(spectrogram.shape) == 3
            spectrogram = np.squeeze(spectrogram, axis=-1)
        log_spec = np.log(spectrogram.T + np.finfo(float).eps)
        height = log_spec.shape[0]
        width = log_spec.shape[1]
        X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
        Y = [i for i in range(height)]
        
        df_wf = pd.DataFrame(columns=["time","Amplitude"])
        df_wf["time"] = np.arange(0, len(waveform))
        df_wf["Amplitude"] = waveform
        fig = px.line(df_wf, x="time", y="Amplitude", title="Audio waveform", width=800, height=400)
        st.plotly_chart(fig)
        
        # Create a trace for the spectrogram
        trace = go.Heatmap( z=log_spec, x=X, y=Y, colorscale='Viridis', showscale=False)
        data=[trace]
        layout = go.Layout( title="Spectrogram", width=800, height=400) 
        fig = go.Figure(data=data, layout=layout)
        st.plotly_chart(fig)
        
        tensor = tf.convert_to_tensor(spectrogram)
        tensor = tf.expand_dims(tensor, 0)
        tensor = np.array(tensor.numpy())
        
        prediction = MODEL.predict(tensor)
        
        # Bar chart of the prediction for each class
        fig = go.Figure(data=[go.Bar(x=TARGETS, y=tf.nn.softmax(prediction[0]))])
        fig.update_layout(title_text='Prediction for each class', width=800, height=400)
        st.plotly_chart(fig)
        st.markdown('##')
        
        best_pred = TARGETS[np.argmax(prediction[0])]
        # st.write("The model predicts: " + best_pred)
        # display the image of the predicted class
        # the images are stored in assets/classes/
        st.image(IMAGE_PATH + best_pred + ".png", caption=f"The model predicts: {best_pred}", width=400, use_column_width=False)
        st.markdown('##')
        st.markdown('##')
            
    else:
        st.markdown('##')
        st.markdown('##')