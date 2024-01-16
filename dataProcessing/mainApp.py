# streamlit run /Users/Manisha/Documents/MS/SDSU/course/BDA-696/final_project/project/ImageDrivenCaptioningAndCustomMusicRecommendations/dataProcessing/mainApp.py
# Link: https://imagedrivencaptioningandcustommusicrecommendations-bda696.streamlit.app/

import os
from pathlib import Path
import streamlit as st
import tempfile
import numpy as np
import time
from imageProcess import image_processing
from textProcess import text_process
from musicProcess import music_recommendation

st.set_page_config(page_title='PicMelody', layout='centered', page_icon='logo2.png', initial_sidebar_state='auto')
st.title('Image Driven Captioning And Custom Music Recommendations', anchor="PicMelody", help='https://github.com/ManishaMatta/ImageDrivenCaptioningAndCustomMusicRecommendations')
st.caption("BDA-696 Course")
caption_text = 'Image Driven Captioning And Custom Music Recommendations'

st.session_state.setdefault('hashtag_multiselect', [])

if 'generate_caption_button' not in st.session_state:
    st.session_state['generate_caption_button'] = False

if 'generate_music_button' not in st.session_state:
    st.session_state['generate_music_button'] = False

if 'get_song_button' not in st.session_state:
    st.session_state['get_song_button'] = False

if 'caption_radio' not in st.session_state:
    st.session_state['caption_radio'] = ''

if 'hashtag_multiselect' not in st.session_state:
    st.session_state.hashtag_multiselect = []

if 'song_radio' not in st.session_state:
    st.session_state['song_radio'] = ''

@st.cache_data(experimental_allow_widgets=True)
def get_details(caption):
    return music_recommendation(caption)

@st.cache_data(experimental_allow_widgets=True)
def get_cap_details(caption_text):
    return text_process(caption_text)

# st.write(__file__)
# st.write(os.path.realpath(__file__))
# st.write(os.path.dirname(os.path.realpath(__file__)))
# st.write(Path.cwd())
# st.write(Path.iterdir("."))
# from pathlib import Path
# st.write([i for i in Path(".").iterdir() if i.is_file()])
# st.write([i for i in Path("/mount/src/imagedrivencaptioningandcustommusicrecommendations/resources/output/").iterdir() if i.is_file()])
# st.write([i for i in Path("/mount/src/imagedrivencaptioningandcustommusicrecommendations/resources/model/").iterdir() if i.is_file()])
# # st.write(os.listdir('/mount/src'))

with st.form(key='image_form'):
    uploaded_file = st.file_uploader("**Choose an Image**")
    if uploaded_file is not None:
        temp_dir = tempfile.mkdtemp()
        path = os.path.join(temp_dir, uploaded_file.name)
        image_uploaded = st.image(uploaded_file)
        output_image_path="/Users/Manisha/Documents/MS/SDSU/course/BDA-696/final_project/project/ImageDrivenCaptioningAndCustomMusicRecommendations/resources/output/"+uploaded_file.name
        with open(os.path.join(output_image_path), "wb") as f:
            f.write(uploaded_file.getbuffer())
        caption_text = image_processing(output_image_path)
    generate_caption_button = st.form_submit_button(label='Generate Captions')
    if generate_caption_button:
        st.session_state['generate_caption_button'] = True


if st.session_state['generate_caption_button']:
    with st.form(key='caption_form'):
        label = get_cap_details(caption_text)
        caption_radio_val = st.radio(
            '**Captions Generated**',
            key="caption_radio",
            options=label[0])
        hash_multiselect = st.multiselect(
            '**Hashtags**',
            label[1], key='hashtag_multiselect')
        generate_music_button = st.form_submit_button(label='Generate Music')
        if generate_music_button:
            st.session_state['generate_music_button'] = True

if st.session_state['generate_music_button']:
    with st.form(key='music_form'):
        caption_dict = get_details(caption_text)
        song_radio = st.radio("**Music Recommended**",  options=[tn['track_name']+" : "+tn['album_name'] for tn in caption_dict], key="song_radio")
        get_song_button = st.form_submit_button(label='Get Song')
        if get_song_button:
            st.session_state['get_song_button'] = True

if st.session_state['get_song_button']:
    with st.form(key='song_form'):
        track_dtls = [tn for tn in caption_dict if tn['track_name'] == song_radio.split(":")[0].strip()][0]
        st.write("**Song Details**")
        st.write("**Song :** "+track_dtls['track_name'])
        st.write("**Album :** "+track_dtls['album_name'])
        st.write("**Artist :** "+track_dtls['artist_names'])
        st.write("**Genres :** " + ('None' if track_dtls['track_genres'].strip() == '' else track_dtls['track_genres']))
        if ((track_dtls['track_preview_url'] is not np.NAN) and (str(track_dtls['track_preview_url']) != 'nan')):
            st.audio(track_dtls['track_preview_url'], format='audio/mp3', start_time=0)
        else:
            st.markdown(f"[{track_dtls['track_link_spotify']}]({track_dtls['track_link_spotify']})")
        details_button = st.form_submit_button(label='Consolidate')
        if details_button:
            st.balloons()
            st.markdown("""<style>.big-font {font-size:20px !important;}</style>""", unsafe_allow_html=True)
            st.markdown(f'<p class="big-font"><b>Caption</b>: {caption_radio_val}</p>'
                        f'<p class="big-font"><b>HashTags</b>: {", ".join(["#"+i for i in hash_multiselect])}</p>'
                        f'<p class="big-font"><b>Song</b>: {song_radio.split(":")[0].strip()}</p>', unsafe_allow_html=True)

with st.spinner('Executing...'):
    time.sleep(1)
st.success('Done!')
