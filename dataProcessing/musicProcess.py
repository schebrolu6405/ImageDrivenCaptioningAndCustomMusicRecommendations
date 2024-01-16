# python /Users/Manisha/Documents/MS/SDSU/course/BDA-696/final_project/project/ImageDrivenCaptioningAndCustomMusicRecommendations/dataProcessing/musicProcess.py

import re
from pathlib import Path
import numpy as np
import requests
import json
import subprocess
import pandas as pd
import time
from builtins import *
from bs4 import BeautifulSoup
from unidecode import unidecode
import nltk
from wordcloud import WordCloud
import gensim
from gensim.utils import simple_preprocess
import gensim.corpora as corpora
from sklearn.neighbors import NearestNeighbors
from datetime import datetime
from nltk.corpus import stopwords
import ast
from common import CommonModule
import warnings

# nltk.download('stopwords')
warnings.filterwarnings("ignore")
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)


class MusicModule:
    ACCESS_TOKEN = ''
    TOKEN_TIME = ''
    LYRICS_TOPICS = ''
    LDA_MODEL = ''
    id2word = ''
    CURRENT_DATE = datetime.now().strftime("%Y%m%d")

    @staticmethod
    def generate_bearer_token():
        URL = "https://accounts.spotify.com/api/token"
        HEADER = {'Content-Type': 'application/x-www-form-urlencoded'}
        # PARAMS = {"grant_type": "client_credentials",
        #           "client_id": "9d7429fddef847139c8ae837b6bcdd92",
        #           "client_secret": "75bef32687344efa8a6f2a7e2b4ce132"
        #           }  # Pic-Melody SDSU id
        PARAMS = {"grant_type": "client_credentials",
                  "client_id": "1bf9921c903e46c2840cc3a8e6c2a08f",
                  "client_secret": "1cb91975733d4d3f88df044d0cbd460a"
                  }  # personal id
        MusicModule.TOKEN_TIME = datetime.now().strftime("%Y%m%d%H%M%S")
        response = requests.post(URL, headers=HEADER, data=PARAMS)
        if response.status_code == 200:
            print("Request was successful")
            MusicModule.ACCESS_TOKEN = response.json().get('access_token')
        else:
            print("Request failed with status code:", response.status_code)

    @staticmethod
    def validate_access_token():
        if MusicModule.ACCESS_TOKEN == '' or (datetime.now().strftime("%Y%m%d%H%M%S") - MusicModule.TOKEN_TIME) >= 3600:
            return False
        else:
            return True

    @staticmethod
    def spotify_trendy_track(url):
        time.sleep(10)
        command = f"curl --request GET --url '{url}' --header 'Authorization: Bearer {MusicModule.ACCESS_TOKEN}'"
        result = subprocess.check_output(command, shell=True, text=True)
        if 'error' in list(json.loads(result).keys()):
            MusicModule.generate_bearer_token()
            MusicModule.spotify_trendy_track(url)
        return result

    @staticmethod
    def is_valid_link(url, retry_url=1):
        try:
            response = requests.head(url)
            if response.status_code < 400:
                return url
            else:
                if retry_url == 1:
                    link_nval = url.lower().split('-sped-')
                    if len(link_nval) > 1:
                        new_url = link_nval[0] + '-lyrics'
                        return MusicModule.is_valid_link(new_url, 2)
                    else:
                        return MusicModule.is_valid_link(url, 2)
                elif retry_url == 2:
                    link_nval = url.lower().split('-from-')
                    if len(link_nval) > 1:
                        new_url = link_nval[0] + '-lyrics'
                        return MusicModule.is_valid_link(new_url, 3)
                    else:
                        return MusicModule.is_valid_link(url, 3)
                elif retry_url == 3:
                    link_nval = url.lower().split('-with-')
                    if len(link_nval) > 1:
                        new_url = link_nval[0] + '-lyrics'
                        return MusicModule.is_valid_link(new_url, 4)
                    else:
                        return MusicModule.is_valid_link(url, 4)
                elif retry_url == 4:
                    link_nval = url.lower().split('-w-')
                    if len(link_nval) > 1:
                        new_url = link_nval[0] + '-lyrics'
                        return MusicModule.is_valid_link(new_url, 5)
                    else:
                        return MusicModule.is_valid_link(url, 5)
                elif retry_url == 5:
                    link_nval = url.lower().split('-single-')
                    if len(link_nval) > 1:
                        new_url = link_nval[0] + '-lyrics'
                        return MusicModule.is_valid_link(new_url, 6)
                    else:
                        return MusicModule.is_valid_link(url, 6)
                elif retry_url == 6:
                    link_nval = url.lower().split('-club-')
                    if len(link_nval) > 1:
                        new_url = link_nval[0] + '-lyrics'
                        return MusicModule.is_valid_link(new_url, 7)
                    else:
                        return MusicModule.is_valid_link(url, 7)
                elif retry_url == 7:
                    link_nval = url.lower().split('-2011-')
                    if len(link_nval) > 1:
                        new_url = link_nval[0] + '-lyrics'
                        return MusicModule.is_valid_link(new_url, 8)
                    else:
                        return MusicModule.is_valid_link(url, 8)
                elif retry_url == 8:
                    if url.find('&') != -1:
                        new_url = url.lower().replace('&', 'and')
                        return MusicModule.is_valid_link(new_url, 9)
                    else:
                        return MusicModule.is_valid_link(url, 9)
                else:
                    return url
        except requests.exceptions.RequestException:
            print("exception", retry_url)
            return url

    @staticmethod
    def lyric_url(lyric_url):
        if lyric_url == '':
            print("***** lyrics not found *****")
        else:
            pass
        if type(lyric_url) is str:
            return unidecode(((lyric_url.split('(')[0] + lyric_url.split(')')[1] if (
                    len(lyric_url.split(')')) > 1 and len(lyric_url.split(')')[1]) > 0) else
                               lyric_url.replace('(', '').replace(')', ''))).split('[')[0].split("feat.")[0].strip()
                             .replace(",", "-and-").replace("/", "").replace(".", "").replace("\'", "").replace(" ", "-")
                             .replace("’", "").replace("--", "-").replace('--', '-').replace('?', '').replace('!', ''))
        else:
            return unidecode(
                "-".join("-and-".join(lyric_url).strip().split()).replace(".", "").replace(",", "").replace('\'', '')
                .replace('--', '-').replace('?', '').replace('!', ''))

    @staticmethod
    def spotify_features(row_dict, track_id):
        url = "https://api.spotify.com/v1/audio-features?ids=" + track_id
        data = json.loads(MusicModule.spotify_trendy_track(url))
        try:
            row_dict['track_danceability'] = data['audio_features'][0]['danceability']
            row_dict['track_energy'] = data['audio_features'][0]['energy']
            row_dict['track_loudness'] = data['audio_features'][0]['loudness']
            row_dict['track_speechiness'] = data['audio_features'][0]['speechiness']
            row_dict['track_acousticness'] = data['audio_features'][0]['acousticness']
            row_dict['track_instrumentalness'] = data['audio_features'][0]['instrumentalness']
            row_dict['track_liveness'] = data['audio_features'][0]['liveness']
            row_dict['track_tempo'] = data['audio_features'][0]['tempo']
            row_dict['track_valence'] = data['audio_features'][0]['valence']
            row_dict['track_track_href'] = data['audio_features'][0]['track_href']
        except Exception as e:
            print(data)
            print(e)

    @staticmethod
    def spotify_artist_genres(row_dict, artist_id):
        genre_list = []
        for a_id in artist_id:
            url = "https://api.spotify.com/v1/artists/" + a_id
            data = json.loads(MusicModule.spotify_trendy_track(url))
            if 'genres' in list(data.keys()):
                genre_list += data['genres']
        row_dict['track_genres'] = genre_list

    @staticmethod
    def spotify():
        url = "https://api.spotify.com/v1/playlists/6QGk7b8naF3ZPljgPtWMAD"
        MusicModule.generate_bearer_token()
        if not MusicModule.validate_access_token:
            MusicModule.generate_bearer_token()

        data = json.loads(MusicModule.spotify_trendy_track(url))
        base_row_dict = {}
        data_list = [data]
        base_row_dict['playlist_name'] = data['name']
        base_row_dict['playlist_followers'] = data['followers']['total']
        base_row_dict['playlist_uri'] = data['uri']
        base_row_dict['playlist_id'] = data['id']
        base_row_dict['total_tracks'] = data['tracks']['total']
        page = data['tracks']['next']
        while str(page) != 'None':
            data = json.loads(MusicModule.spotify_trendy_track(page))
            data_list.append(data)
            page = data['next']

        rows = []
        for i in data_list:
            if ('tracks' in i.keys()) and ('items' not in i.keys()):
                for j in i['tracks']['items']:
                    row_dict = base_row_dict.copy()
                    row_dict['track_added_time'] = j['added_at']
                    row_dict['track_id'] = j['track']['id']
                    row_dict['artist_names'] = [nm['name'] for nm in j['track']['album']['artists']]
                    row_dict['artist_id'] = [nm['id'] for nm in j['track']['album']['artists']]
                    row_dict['album_available_market'] = j['track']['album']['available_markets']
                    row_dict['album_name'] = j['track']['album']['name']
                    row_dict['album_type'] = j['track']['album']['type']
                    row_dict['album_release_date'] = j['track']['album']['release_date']
                    row_dict['album_date_precision'] = j['track']['album']['release_date_precision']
                    row_dict['album_tracks'] = j['track']['album']['total_tracks']
                    row_dict['track_available_market'] = j['track']['available_markets']
                    row_dict['track_duration_ms'] = j['track']['duration_ms']
                    row_dict['track_link_spotify'] = j['track']['external_urls']['spotify']  # IMP: link of the song
                    row_dict['track_name'] = j['track']['name']
                    row_dict['track_popularity'] = j['track']['popularity']
                    row_dict['track_preview_url'] = j['track']['preview_url']  # IMP : play song directly
                    MusicModule.spotify_features(row_dict, j['track']['id'])
                    MusicModule.spotify_artist_genres(row_dict, row_dict['artist_id'])
                    MusicModule.sentiment_lyrics(row_dict)
                    rows.append(row_dict)
            else:
                for j in i['items']:
                    row_dict = base_row_dict.copy()
                    row_dict['track_added_time'] = j['added_at']
                    row_dict['track_id'] = j['track']['id']
                    row_dict['artist_names'] = [nm['name'] for nm in j['track']['album']['artists']]
                    row_dict['artist_id'] = [nm['id'] for nm in j['track']['album']['artists']]
                    row_dict['album_available_market'] = j['track']['album']['available_markets']
                    row_dict['album_name'] = j['track']['album']['name']
                    row_dict['album_type'] = j['track']['album']['type']
                    row_dict['album_release_date'] = j['track']['album']['release_date']
                    row_dict['album_date_precision'] = j['track']['album']['release_date_precision']
                    row_dict['album_tracks'] = j['track']['album']['total_tracks']
                    row_dict['track_available_market'] = j['track']['available_markets']
                    row_dict['track_duration_ms'] = j['track']['duration_ms']
                    row_dict['track_link_spotify'] = j['track']['external_urls']['spotify']  # IMP: link of the song
                    row_dict['track_name'] = j['track']['name']
                    row_dict['track_popularity'] = j['track']['popularity']
                    row_dict['track_preview_url'] = j['track']['preview_url']  # IMP : play song directly
                    MusicModule.spotify_features(row_dict, j['track']['id'])
                    MusicModule.spotify_artist_genres(row_dict, row_dict['artist_id'])
                    MusicModule.sentiment_lyrics(row_dict)
                    rows.append(row_dict)
        return pd.DataFrame(rows)

    @staticmethod
    def pre_process(lyrics):
        if isinstance(lyrics, str):
            cleaned_lyrics = lyrics.strip().replace('\d+', '')
            cleaned_lyrics = re.sub('[,\.!?]', '', cleaned_lyrics.lower())
            return cleaned_lyrics
        else:
            print(lyrics)

    @staticmethod
    def remove_stopwords(texts):
        return [[word for word in simple_preprocess(str(doc)) if word not in MusicModule.stopwords()] for doc in texts]

    @staticmethod
    def sent_to_words(sentences):
        for sentence in sentences:
            yield gensim.utils.simple_preprocess(str(sentence), deacc=True)

    @staticmethod
    def lda_lyrics(spotify_df):
        spotify_df.dropna(subset=['track_lyrics'])
        spotify_df['track_lyrics_processed'] = spotify_df['track_lyrics'].apply(MusicModule.pre_process)
        spotify_df['track_lyrics_processed_list'] = spotify_df['track_lyrics_processed'].apply(
            lambda x: MusicModule.remove_stopwords(list(MusicModule.sent_to_words([x]))))
        spotify_df['track_lyrics'] = spotify_df['track_lyrics'].str.replace('\n', ' ').str.replace('\s+', ' ',regex=True)
        spotify_df['track_lyrics_processed'] = spotify_df['track_lyrics_processed'].str.replace('\n', ' ').str.replace('\s+', ' ', regex=True)

    @staticmethod
    def lda_model_lyrics(spotify_df):
        data = spotify_df['track_lyrics_processed'].values.tolist()
        data_words = list(MusicModule.sent_to_words(data))
        data_words = MusicModule.remove_stopwords(data_words)
        MusicModule.id2word = corpora.Dictionary(data_words)
        texts = data_words
        corpus = [MusicModule.id2word.doc2bow(text) for text in texts]
        lda_model = gensim.models.LdaModel(corpus=corpus, id2word=MusicModule.id2word, num_topics=10)
        MusicModule.LDA_MODEL = lda_model
        MusicModule.LYRICS_TOPICS = lda_model.print_topics()
        spotify_df['Bow_Vector'] = spotify_df['track_lyrics_processed_list'].apply(
            lambda x: [MusicModule.id2word.doc2bow(j) for j in ast.literal_eval(x)])  # eval(x)])  # ast.literal_eval(x)])
        spotify_df['lyrics_topic_distribution'] = spotify_df['Bow_Vector'].apply(lambda x: lda_model[x])

    @staticmethod
    def stopwords():
        # @TODO Wordcloud: display for report and store it in first run
        stop_words = stopwords.words('english')
        stop_words.extend(
            ['i', 'i\'m', 'hii', 'hi', 'might', 'even', 'got', 'ooh', 'oh', 'wanna', 'na', 'yeah', 'would', 'from',
             'to', 'do', 'be', 'in', 'for', 'my', 'how', 'of', 'get', 'know', 'uh', 'ya', 'like', 'iz'])
        return stop_words

    @staticmethod
    def lyrics_wordcloud(spotify_df):
        long_string = ','.join([lyric for lyric in list(spotify_df['track_lyrics_processed']) if type(lyric) == str])
        wordcloud = WordCloud(stopwords=MusicModule.stopwords(), background_color="white", max_words=500,
                              contour_width=3, contour_color='steelblue').generate(long_string)
        wordcloud.to_file("/Users/Manisha/Documents/MS/SDSU/course/BDA-696/final_project/project"
                          "/ImageDrivenCaptioningAndCustomMusicRecommendations/resources/pictures/lyrics_wordcloud.png")

    @staticmethod
    def sentiment_lyrics(row_dict):
        row_dict['artists'] = MusicModule.lyric_url(row_dict['artist_names'])  # .apply(MusicModule.lyric_url)
        row_dict['track_names'] = MusicModule.lyric_url(row_dict['track_name'])  # .apply(MusicModule.lyric_url)
        base_url = 'https://genius.com/'
        end_url = 'lyrics'
        row_dict['artists_track'] = base_url + row_dict['artists'] + '-' + row_dict['track_names'] + '-' + end_url
        lyrics = []
        new_url = MusicModule.is_valid_link(row_dict['artists_track'])
        time.sleep(10)
        response = requests.get(new_url)
        if response.status_code == 200:
            content = BeautifulSoup(response.text, 'html.parser')
            for val in content.find_all("span", class_="ReferentFragmentdesktop__Highlight-sc-110r0d9-1 jAzSMw"):
                lyrics.append(val.text)
            track_lyrics = '\n'.join(lyrics)
            row_dict['track_lyrics'] = track_lyrics
            row_dict['track_lyrics_sentiment'] = CommonModule.sentiment(track_lyrics)
        else:
            print("Invalid url", new_url)
            row_dict['track_lyrics'] = np.NAN
            row_dict['track_lyrics_sentiment'] = np.NAN

    @staticmethod
    def lyrics(input_df):
        input_df['artists'] = input_df['artist_names'].apply(MusicModule.lyric_url)
        input_df['track_names'] = input_df['track_name'].apply(MusicModule.lyric_url)
        input_df['base_url'] = 'https://genius.com/'
        input_df['end_url'] = 'lyrics'
        input_df['artists_track'] = input_df['base_url'].str.cat(
            input_df['artists'].str.cat(input_df['track_names'], sep='-')).str.cat(input_df['end_url'], sep='-')
        urls = input_df['artists_track']
        track_lyrics = {}
        for url in urls:
            lyrics = []
            new_url = MusicModule.is_valid_link(url)
            response = requests.get(new_url)
            if response.status_code == 200:
                content = BeautifulSoup(response.text, 'html.parser')
                for val in content.find_all("span", class_="ReferentFragmentdesktop__Highlight-sc-110r0d9-1 jAzSMw"):
                    lyrics.append(val.text)
                track_lyrics[url] = ' '.join(lyrics)
            else:
                print("Invalid url", url)
                track_lyrics[url] = url
        return track_lyrics

    @staticmethod
    def join_lyrics_songs():
        # file_path = Path(f"resources/datasets/music_{current_date}_0_11.csv")
        file_path_spotify = Path("/Users/Manisha/Documents/MS/SDSU/course/BDA-696/final_project/project/ImageDrivenCaptioningAndCustomMusicRecommendations/resources/datasets/music_tracks.csv")

        if not file_path_spotify.is_file():
            spotify_df = MusicModule.spotify()
            MusicModule.lda_lyrics(spotify_df)
            spotify_df.to_csv(file_path_spotify, sep='|', header='true', index=False)
        else:
            spotify_df = pd.read_csv(file_path_spotify, sep='|', header='infer', skipinitialspace=True)
            MusicModule.lyrics_wordcloud(spotify_df)  # comment this after 1st execution
        return spotify_df

    @staticmethod
    def nearest_neighbors(dataframe, column, target_value, neighbors=20):
        if neighbors > len(dataframe) > 0:
            neighbors = len(dataframe)
        elif len(dataframe) == 0:
            return dataframe
        knn_model = NearestNeighbors(n_neighbors=neighbors)
        knn_model.fit(dataframe[column].values.reshape(-1, 1))
        distances, indices = knn_model.kneighbors([[target_value]])
        knn_df = dataframe[column].iloc[indices[0][:neighbors]]
        knn_df = knn_df.reset_index().rename(
            columns={'index': 'join_index', 'caption_sentiment': 'caption_sentiment_knn'}).reset_index().rename(
            columns={'index': 'knn_order'})
        return pd.merge(dataframe, knn_df, left_index=True, right_on='join_index')

    @staticmethod
    def caption_parser(spotify_df, caption):
        # LDA group
        MusicModule.lda_model_lyrics(spotify_df)
        bow = MusicModule.id2word.doc2bow(caption.split())
        topic_distribution = MusicModule.LDA_MODEL.get_document_topics(bow)
        most_probable_topic = max(topic_distribution, key=lambda x: x[1])
        # print(f" most probable Topic {most_probable_topic[0]} with probability {most_probable_topic[1]}")
        if round(most_probable_topic[1].item(), 2) > 0.1:
            spotify_df = spotify_df[spotify_df["lyrics_topic_distribution"]
                                    .map(lambda x: max(x[0], key=lambda y: y[1])[0]) == most_probable_topic[0]]
        # similarity test for the lyrics based on caption
        spotify_df['track_similarity_score'] = spotify_df['track_lyrics_processed'].map(lambda x: CommonModule.similarity_score(caption.strip(), x.strip()) if type(x) == str else 0.0)
        spotify_df_filter_lda = spotify_df.sort_values(by=["track_similarity_score"], ascending=[False]).head(30)
        # sentiment group
        senti = CommonModule.sentiment(caption)
        caption_senti = round(senti['pos'] - senti['neg'], 2)
        spotify_df_filter_lda["caption_sentiment"] = spotify_df_filter_lda["track_lyrics_sentiment"].map(
            lambda x: round(float(ast.literal_eval(x)['pos']) - float(ast.literal_eval(x)['neg']), 2) if pd.notna(
            # lambda x: round(float(eval(x)['pos']) - float(eval(x)['neg']), 2) if pd.notna(
                x) else 0)
        spotify_df_filter_senti = MusicModule.nearest_neighbors(spotify_df_filter_lda, "caption_sentiment",
                                                                caption_senti)
        if len(spotify_df_filter_senti) == 0:
            spotify_df_filter_senti = spotify_df_filter_lda
        # order by  sequences
        spotify_df_filter_senti = spotify_df_filter_senti.sort_values(
            by=["track_popularity", "track_valence", "track_danceability", "track_energy"],
            ascending=[False, False, False, True])
        # sequencing the best 5 tracks for the requirements
        display_music = spotify_df_filter_senti[
            ['track_name', 'album_name', 'artist_names', 'track_preview_url', 'track_link_spotify', 'track_genres']].head(5)
        display_music['artist_names'] = display_music['artist_names'].map(lambda x: ','.join(ast.literal_eval(x)))
        display_music['track_genres'] = display_music['track_genres'].map(lambda x: ','.join(ast.literal_eval(x)))
        return display_music.to_dict(orient='records')


def music_recommendation(caption):
    spotify_df = MusicModule.join_lyrics_songs()
    return MusicModule.caption_parser(spotify_df, caption)


# print(datetime.now())
# caption_text1 = "A little girl climbing the stairs to her playhouse"
# caption_text2 = "Four men on top of a tall structure"
# print(music_recommendation(caption_text1+caption_text2))
# print(datetime.now())

# music_recommendation("Two young guys with shaggy hair look at their hands while hanging out in the yard .")
