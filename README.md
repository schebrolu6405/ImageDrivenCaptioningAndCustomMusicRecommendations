# ImageDrivenCaptioningAndCustomMusicRecommendations
BDA696: Image-Driven Captioning and Custom Music Recommendations (final project)

### Members
@ManishaMatta
@Sabnam-Pandit
@chebrolu6405
@anurima-saha



### Objective
The project aims to integrate captions and music with images to enhance social networking experiences. 
We aim to develop an Image Captioning and Music Recommendation System which accepts images as input and produces platform suitable captions and music as output. 
Our approach involves harnessing computer vision algorithms along with language models using Tensorflow in Python to create generalized image descriptions that portray the actions(verbs), emotions(adjectives), and contexts associated with the depicted objects(nouns). 
Further refinement is performed by incorporating Natural Language Processing while scraping data from multiple web sources to generate appropriate captions and hashtags. 
Additionally, we are integrating a music recommendation engine into the system. This engine will analyze the generated captions and offer suggestions of trending music that aligns with overall essence in the images.

### Technical Overview
[Flow Diagram](https://lucid.app/lucidchart/8a5c5cb6-197e-4da1-a721-9a6f9ca56cb3/edit?invitationId=inv_0c7bd8ad-3f34-4930-8059-94764fd01486&page=0_0#)
![Pic-Me-lody.png](resources%2Fpictures%2Fdoc%2FPic-Me-lody.png)

##### 1. VGG Model
A VGG16 uses convolutional layers with filter size 3x3 with padding 2x2 and a stride of 2, along with 2x2 max pooling layers having a stride of 2. The detailed architecture of how the model is being used for image compression is described below:
* Conv1: In conv1 we have used two convolutional layers that convert a 224*224*3 image into 224*224*64. The following max pooling reduces features to 112*112*64.
* Conv2: In conv2 we use two convolution.It converts 112*112*64 features to 112*112*128. After that it does the max pooling which converts features to 56*56*128
* Conv3: In conv3 we use three convolution.It converts features into 56*56*256. After that it does the max pooling which converts features to 28*28*256
* Conv4: In conv4 we use three convolutions.It converts features into 28*28*512. After that it does the max pooling which converts features to 14*14*512
* Conv5: In conv5 we use three convolution.It converts features into 14*14*512. After that it does the max pooling which converts features to 7*7*512
fc6 & fc7: These fully connected layers give a final output of 4096 features.

##### 2. LSTM Model
Long Short Term Memory networks(LSTM), introduced by Hochreiter & Schmidhuber (1997), are a special kind of RNN, capable of learning long-term dependencies .  
Given the goal of obtaining words from images,there are two primary advantages of using LSTM in our project:
* Learning based on bounding boxes to label mapping finds it difficult to learn abstract concepts like “beautiful” or “open” or “crowded”, which are very significant for generating our final results.
* Training LSTM helps us capture common sense ideas better. For example, it does not simply classify an object(image of a glass containing orange liquid) as “glass” and/or “orange” but as “orange juice”.
The expectation from using LSTM model along with fully connected layers for image features is to capture the joint multimodal distribution of images and their text representation to generate a generalized description containing adjectives,nouns and verbs which are of key importance to fully comprehend the true sense of the image as necessary for our final captioning and music recommendation.

##### 3. Transformer model
Transformer model is used to perform various natural language processing (NLP) tasks and other sequence-to-sequence tasks in machine learning. It is the self-attention mechanism, which allows the model to weigh the importance of different words in a sequence when making predictions. This mechanism enables the model to capture long-range dependencies and relationships between words, making it particularly effective for tasks involving sequential data, such as text summarization, and tags generation.
Some of the key components of the transformer model are Self Attention which computes a score to each keyword of the sentence and these scores will determine how much focus the model should be given to each word when making predictions. This model has an advantage to use multiple sets of scores and outputs from those sets concatenated to form a refined output. This multiple sets processes with encoder and decoder which consists of multiple layers which contain multiple heads and self-attention mechanism where encoder processes input sequence and decoder processes output.  
Transformer architecture has proven to be highly effective and scalable, enabling the development of large pre-trained models like BERT, GPT, and T5, which have achieved state-of-the-art performance on various NLP tasks.

##### 4. Topic Modeling using Latent Dirichlet Allocation (LDA), gensim library in python 
LDA assumes that documents are mixtures of topics and that each word in a document is attributable to one of the document's topics.
* Initialization: LDA starts by assuming a fixed number of topics for the entire corpus and a distribution of topics for each document. The number of topics is a hyperparameter that needs to be set prior to training.
* Assigning Topics: For each document in the corpus, LDA assigns a distribution of topics based on the words in the document. Each word in the document is then assigned a specific topic based on a probability distribution.
* Generative Process: LDA assumes a generative process for creating documents, For each document, choose a distribution of topics. For each word in the document, Choose a topic from the distribution of topics or Choose a word from the topic's distribution of words.
* Parameter Estimation: The goal of training an LDA model is to estimate the parameters (topic distributions for documents, word distributions for topics) that best explain the observed documents.
* Dirichlet Priors: LDA uses Dirichlet priors to model the distribution of topics and words. The Dirichlet distribution is a family of continuous multivariate probability distributions.
* Inference: Inference in LDA involves estimating the posterior distribution of topics for each document and the distribution of words for each topic. This is typically done using techniques like variational inference or Gibbs sampling.
The output of LDA includes the identified topics and the words associated with each topic. Each document is represented as a mixture of topics, and each topic is represented as a distribution of words.

##### 5. Similarity Scoring Model using TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer and cosine similarity, sk-learn library in python
To find similar documents based on textual content
* TF-IDF Vectorization: The TfidfVectorizer from scikit-learn is used to convert the preprocessed text data into numerical vectors using the TF-IDF representation. TF-IDF reflects the importance of a word in a document relative to its importance in the entire dataset.
* Cosine Similarity Calculation: The cosine_similarity function from scikit-learn computes the cosine similarity between documents based on their TF-IDF vectors. Cosine similarity measures the cosine of the angle between two non-zero vectors, providing a similarity score between 0 and 1.
* Similarity Scoring Function: The get_similarity_scores function takes a query text and calculates the cosine similarity scores between the query and all documents in the dataset. It returns a dictionary where the keys are document texts, and the values are their similarity scores with the query.

##### 6. Sentiment Analysis using SentimentIntensityAnalyzer, Natural Language Toolkit (nltk) library in Python 
It is the process of determining the sentiment or emotional tone expressed in a piece of text.
* Intensity Scores: The SentimentIntensityAnalyzer assigns intensity scores to the text based on the presence of positive, negative, and neutral words, as well as the overall polarity of the text.
* Polarity Score: The tool calculates a polarity score for the given text, which indicates the overall sentiment as positive, negative, or neutral. The polarity score is a continuous value between -1 (most negative) and 1 (most positive).
* Compound Score: The compound score is a single metric that represents the overall sentiment of the text. It is a combination of the positive, negative, and neutral scores, normalized to a range between -1 and 1. A positive compound score suggests a positive sentiment, while a negative score suggests a negative sentiment.
* Use of Lexicons: The SentimentIntensityAnalyzer relies on pre-built lexicons or dictionaries that contain lists of words associated with different sentiments. These lexicons are used to match words in the input text and assign sentiment scores.

##### 7. K-Nearest Neighbors [K-NN], sk-learn library in python 
This is a supervised machine learning algorithm used for classification and regression tasks. It operates by finding the 'k' training examples in its vicinity and assigns the majority class (for classification) or average value (for regression) as the predicted output for a given input.
* Distance Calculation: When a new, unseen data point is to be classified or predicted, KNN calculates the distance between this point and all the points in the training set. The most common distance metric is Euclidean distance, but other metrics can be used based on the nature of the data.
* K Nearest Neighbors: KNN then identifies the 'k' training examples that are closest to the new data point in terms of distance. The value of 'k' is a user-defined parameter.
* Training: It uses a labeled dataset to train the KNN model. The features used are the sentiment labels (positive, negative, neutral).
* Prediction: While passing in the new document(caption). We can then identify which texts are more related based on the predicted labels.

#### I. Image Processing
For feature extraction for images and text we have used the Flickr30k dataset from Kaggle. It has 31 thousand image entities matched with 158 thousand captions on the basis of image ids.
##### Image Pre-Processing and Feature Extraction:
We have downloaded the Flickr30k image data directly from Kaggle using the kaggle api    available in python. This data contained corrupt files which have been removed by:
* Filtering on for files with extension '.jpg', '.jpeg', '.png', '.gif'
* Use try: and except: to identify and remove the corrupt the files from training data
* Removed data from which image dimensions are 0
Using the pre-processed data, we have trained a VGG16 model to extract the features of the image. The test-train split was made as follows:
* Train data - 80%
* Test data - 10%
* Validation data - 10%
VGG16:
The images were resized to 224x224 pixels along with 3 channels (RGB) to format it as an input for the VGG16 model. A batch size of 10 was used to preprocess the images using the preprocess_input function from tensorflow.keras.applications.vgg16 module.
##### Text Pre-Processing and Feature Extraction:
We are loading captions and image ids from the .csv file to map to corresponding images. We clean the data by:
* Converting each caption to lower case
* Remove all blanks in the caption
* Retaining only characters and removing all numbers and special characters.
* Tokenizing  the captions to prepare data for the model. It is using the keras inbuilt  class to build a new tokenizer object Tokenizer(). This tokenizer fits all the captions to create a dictionary.
LSTM:
The model consists of two parts: Encoder and Decoder. The encoder model takes two inputs one is image feature and the other is word sequence. The encoder (consists of fully connected layers and LSTM) converts both image feature and word sequence in 256 dimensions vector.
The decoder adds those vectors and converts them into a vector of vocab_size by using a couple of fully connected layers.
##### Training and Inference:
During training, the model takes two inputs: caption and image. The image goes through VGG and converts into a 4096 dimensional vector.The caption is divided into multiple pairs of in-sequence and next word.The in-sequence goes through the model and the  outcome of which is  matched with the actual next word.Using the loss the model is trained.We identify the best model using minimum validation  loss.       
In inference, we start with in-sequence as <startsequence>    which is passed through the model along with the image features received from VGG.The model predicts the next word.Once ,the next word is predicted we append it to our caption input as a new sequence.We repeat the process to obtain the complete caption of given length.

#### II. Caption Generation
Natural Language Processing (NLP) is harnessed to generate engaging social media captions, so we have considered the Natural Language Toolkit (NLTK) in Python. We embark on a journey of text analytics to transform ordinary captions into compelling narratives. NLTK's robust suite of tools empowers us to tokenize, tag parts of speech, and provide invaluable insights into the linguistic nuances of the text. The primary goal is to craft captivating captions by mining relevant quotes from a designated webpage. Through a blend of web scraping and linguistic analysis. We are seeking to transform plain text into compelling narratives, ensuring a resonance with the social media audience.
* Processing keywords
Now, we process the image description to extract meaningful keywords. We have used NLTK's tokenizer to break down text into words. Additionally, it employs part-of-speech tagging and filters the words based on their POS tags, retaining only verbs (VB), nouns (NN), and adjectives (JJ). Next, we are performing stemming using nltk stem package to remove the suffixes like ed, ing, etc. of the repeated words and leveraging NLTK's WordNet, a lexical database, the code explores synonyms for key words, expanding the range of potential quotes.
* Web Scraping: 
Considering the above list of keywords which will consist of keywords extracted from sentences and their synonyms, we perform a web scraping task to generate a list of relevant captions based on a set of keywords. Initially, we used Beautiful Soup to scrape quotes from these websites (Lifewire.com, Oberlo.com, Goodreads.com).  The scraping function is designed to handle paginated content, continuously extracting captions by utilizing the 'requests' library to make an HTTP request to the provided URL. Upon receiving a successful response (status code 200), the code uses the 'Beautiful Soup' library for HTML parsing. It specifically targets given <tag> elements with a particular class ('QUOTES_CLASS') within the HTML structure. For each identified <tag>, the code further extracts all required tags and appends the text content of each tag to a list. Finally, the function returns the compiled list of quotes until there are no more pages to fetch. The resulting list of captions is stored in another list.
* Ranking Captions: 
This part involves a key strategy of ranking the scraped captions based on the number of matching keywords extracted from the input list. This process aims to prioritize captions that align more closely with the provided keywords, resulting in a curated selection that resonates with the intended context. Machine learning models, particularly cosine similarity check, can play a pivotal role in this ranking mechanism.
* Generate Hashtags: 
In the process of hashtag generation, a curated list of keywords serves as the foundation, this keyword list is systematically expanded by incorporating synonyms associated with each keyword, Synonyms provide variations of a word, ensuring a diverse range of expressions. Incorporating a pre-trained transformers model (t5-base-tag-generation, AutoModelSeq2SeqLM) to generate the tags that are relevant to the image description. The resulting set of synonyms and tags will be formed as another list from where few relevant keywords will be chosen for representation of hashtags.
* Generate Caption: 
Once the top-ranked captions are identified, the corresponding set of hashtags is extracted. These hashtags, representing the core themes and concepts within the captions, are seamlessly integrated with the top-ranked caption to create a cohesive and impactful output.

#### III . Music Recommendation
Our Music Recommendation model operates in two distinct parts. The first part initiates upon triggering the application, running concurrently with other processes. It involves the Music Data extraction system, which collects necessary information and stores it in cache. This cached data is later processed to construct the recommendation model within the job flow. The second part comes into play after the captions are generated from the Text Analytics module. This segment focuses on the analysis and recommendation.
##### Music Data Extraction
The aim of this subsection in our module is to collect necessary data related to currently trending music. Since we utilize web scraping and HTTP API requests to retrieve data for our application, this process can be time-consuming. Therefore, we execute this section at the beginning of our module to mitigate potential delays in generating results.
* HTTP API request for Spotify API: We utilize the Spotify Developer API to obtain an API key along with a client ID and Secret. These credentials are essential for accessing various Spotify APIs.
Upon obtaining the client ID and client Secret, we authenticate a bearer token, which remains active for 60 minutes. Consequently, we generate unique tokens after their expiration during the execution. This token serves the purpose of connecting to the web page and fetching the latest playlist for internationally trending songs. The playlist contains comprehensive details about each song like its artist, album, popularity, and a playable URL. This URL is utilized to showcase the recommended songs.
In addition to the API key and song details, we retrieve the features of all tracks. These features include essential analytical metrics like loudness, danceability, liveliness, and energy of the song. Genre is also collected based on the artist with a similar API call. These collected attributes from Spotify serve as a foundation for our recommendation system.
* Web Scraping with BeautifulSoup from Genius.com API: We now possess all data required for building the recommendation system except for the track lyrics. Therefore, we are using the Genius.com web page. The query URL of this webpage can be modified with the artist, album, and song details to display the song lyrics. Incorporating various string synthesis methods and blended logics based on the track details, we generate the url to retrieve the lyrics. BeautifulSoup Web Scraping method is used to read the lyrics.We preprocess these lyrics into the required format to be added to the existing song details.
##### Music Data Analysis and Recommendation
This block under the music recommendation module is executed post the captions generation. We are considering the captions as the primary input component for our model to analyze song lyrics and genre and recommend a song track suitable for the caption.
* LDA Analyzing Lyrics: We employ Latent Dirichlet Allocation (LDA) topic modeling to analyze song lyrics and captions, selectively identifying songs that align with the analysis. The main goal of this is to identify the essence of the song from their words. We are using the modules from the gensim package for this analysis. First, we tokenize the song lyrics after removing stopwords, creating a dictionary (id2word) and representing them in a bag-of-words format. The Gensim LDAModel is then trained on this corpus to identify 10 topics. The resulting model enables us to deduce the themes of songs and associate each song with its corresponding topic distribution. This information is subsequently employed in the analysis of captions..
* Similarity Analysis: In the next phase of our recommendation model, we employ modules from the sklearn package, such as CountVectorizer and cosine_similarity. CountVectorizer helps transform textual data, such as captions and song lyrics, into numerical vectors. Subsequently, we calculate the cosine similarity between these vectors to quantify the degree of similarity or dissimilarity. The output is a similarity score, with higher values signifying greater similarity and lower values indicating less resemblance.
* Sentiment Analysis: We are capturing the sentiment of the caption and the sentiment of all song lyrics using the SentimentIntensityAnalyzer, this returns if the parameter passed through the analyser is positive, negative or neutral. Additionally, we are integrating the k-Nearest Neighbors (KNN) algorithm to find the nearest neighbors for the caption sentiment based on the lyrical sentiment of the songs. With this information we can understand the tone and classify them accordingly.
* Song Classifier: In addition to analyzing songs into separate groups we are also ranking them, so we can display only the most relevant track for a caption. We are implementing this by reviewing the song features, with this approach we are reorganizing them based on different characteristics and consolidating based on the caption.
* Recommendation Model: From the above analysis we can correlate the captions with the classified track group, and prioritize them based on the feature relevance. Thus returning songs which are suitable for the caption.

### Results
![Outpur_1.png](resources%2Fpictures%2Fdoc%2FOutpur_1.png)
![Output_2.png](resources%2Fpictures%2Fdoc%2FOutput_2.png)
![Output_3.png](resources%2Fpictures%2Fdoc%2FOutput_3.png)

### Thank you
