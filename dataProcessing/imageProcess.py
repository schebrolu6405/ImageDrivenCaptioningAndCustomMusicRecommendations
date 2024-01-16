# python /Users/Manisha/Documents/MS/SDSU/course/BDA-696/final_project/project/ImageDrivenCaptioningAndCustomMusicRecommendations/dataProcessing/imageProcess.py
# tensorflow: 2.12.0
# keras: 2.12.0
# python: 3.11.5

import os  # handle files and directories
import numpy as np
import pickle  # saving and  loading python objects
from tensorflow.keras.models import Model
from tqdm.notebook import tqdm  # ADDs progress bar to loops and iterable object
from tensorflow.keras.preprocessing.image import load_img, img_to_array  # loads the img and return PIL image object then PIL object is converted into numpy array.
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input  # import vgg16 model
from tensorflow.keras.preprocessing.text import Tokenizer  # raw text into a numerical format
from tensorflow.keras.preprocessing.sequence import pad_sequences  # padding sequences to a specified length and optionally truncating longer sequences
from tensorflow.keras.utils import to_categorical, plot_model  # used for one-hot encoding  categorical integer data.plot_model is used for visualizing the architecture
from tensorflow.keras.layers import Input, Dense, Embedding, LSTM, Dropout, add  # gives shape of your model, connects previous layers to current layers,converting categorical data to continous representation,dropout for regularization
from keras.models import load_model  # to load saved lstm model
from PIL import Image
import matplotlib.pyplot as plt
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint, EarlyStopping


class ImageModule:
    # Data Generator function to make it fast and avoid session crash
    @staticmethod
    def data_loader(Images_set, caption_map, features, tokenizer, max_length, vocab_size, batch_size):
        # loop over image

        X1, X2, y = list(), list(), list()
        n = 0
        while 1:
            for Image_id in Images_set:
                n += 1
                captions = caption_map[Image_id]
                # process each caption
                for caption in captions:
                    # encode the sequence
                    seq = tokenizer.texts_to_sequences([caption])[0]
                    # split the sequence into X, y pairs
                    for i in range(1, len(seq)):
                        # split into input and output pairs
                        in_seq, out_seq = seq[:i], seq[i]
                        # pad input sequence
                        in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                        # encode output sequence
                        out_seq = to_categorical([out_seq],
                                                 num_classes=vocab_size)[0]
                        X1.append(features[Image_id])
                        X2.append(in_seq)
                        y.append(out_seq)
                        if i > 3: break
                        if n == batch_size:
                            X1, X2, y = np.array(X1), np.array(X2), np.array(y)
                            yield [X1, X2], y
                            X1, X2, y = list(), list(), list()
                            n = 0
                            # break

    ##Preprocessing Text(Captions) data
    @staticmethod
    def cleaning(caption_map):
        for key, captions in caption_map.items():
            for i in range(len(caption)):
                # one caption at a time
                caption = captions[i]
                # convert to lowercase
                caption = caption.lower()
                # remove special char,number etc.
                caption = caption.replace('^[A-Z,a-z]', '')  # Except A-Z,a-z everything is removed
                # remove spaces
                caption = caption.replace('\s+', ' ')
                # add start and end tags to the caption
                caption = 'startseq ' + " ".join([word for word in caption.split() if len(word) > 1]) + ' endseq'
                captions[i] = caption

    @staticmethod
    def load_features(base_dir):

        dir_feature = os.path.join(base_dir, 'features')  # Saving pkl object in base directory
        with open(os.path.join(dir_feature, 'features.pkl'), 'rb') as f:
            features = pickle.load(f)
        return features

    @staticmethod
    def save_features(base_dir):
        features = {}

        # dir=os.path.join(working_dir,'drive/MyDrive/Collab_Projects/CaptionGenerator/flickr30k_images/flickr30k_images')

        dir = os.path.join(base_dir, 'flickr30k_images/flickr30k_images')

        corrupt_count = 0
        image_files = os.listdir(dir)  # List of images in image_files from the directory

        # List all image files in the directory
        image_files = [f for f in os.listdir(dir) if f.endswith(('.jpg', '.jpeg', '.png', '.gif'))]

        # Batch size (the number of images to process in each batch)
        batch_size = 10

        # Iterate over the image files in batches to read Images
        for i in tqdm(range(0, len(image_files), batch_size), desc='Processing batches'):
            image_list = []
            Imageid_list = []
            batch = image_files[i:i + batch_size]
            for image_name in batch:
                imgpath = dir + '/' + image_name
                try:
                    image = load_img(imgpath, target_size=(224, 224))  # Target size resize the image
                    if image.size[0] == 0 or image.size[1] == 0:
                        print(f"Skipping empty image: {imgpath}")
                        continue

                except:
                    corrupt_count += 1
                    print('Corrupt:', corrupt_count)
                    continue

                # Convert Image Pixels to an array
                image = img_to_array(image)
                # Reshape data for model
                image = image.reshape(1, image.shape[0], image.shape[1], image.shape[
                    2])  # RGB image Batch size=1 as we take one image at a time ,3*224*224 where 3=RGB
                image_list.append(image)

                # Get image id
                Image_id = image_name.split('.')[0]
                Imageid_list.append(Image_id)

            # Concatenate 10 numpy array's in image_list
            image_batch = np.concatenate(image_list, axis=0)

            # Preprocess image for vgg
            image = preprocess_input(image_batch)
            # Extract feature of an image
            feature = ImageModule.model_vgg.predict(image, verbose=0)
            # store feature
            for k, row in enumerate(feature):
                features[Imageid_list[k]] = row

        dir_feature = os.path.join(base_dir, 'features')  # Saving pkl object in base directory
        # Store features in pickle object
        pickle.dump(features, open(os.path.join(dir_feature, 'features.pkl'), 'wb'))
        return

    @staticmethod
    def create_lstm_model(vocab_size, max_length):
        # encoder model
        # image feature layers
        inputs1 = Input(shape=(
        4096,))  # 4096 Diemnsional feature is an input that goes to FC layer with relu activation which converts it into 256 dimensional vector.
        fe1 = Dropout(0.05)(inputs1)
        fe2 = Dense(256, activation='relu')(fe1)
        print(fe2.shape)
        # sequence feature layers
        inputs2 = Input(shape=(max_length,))
        se1 = Embedding(vocab_size, 256, mask_zero=True)(
            inputs2)  # token (of size 18000) is converted to 256 dim vector by Embedding layer. The token is of that length because tokenizer converts each word into that length vector
        se2 = Dropout(0.05)(se1)
        se3 = LSTM(256, return_sequences=True)(se2)  # 256 dim vector is passed through LSTM twice
        se4 = LSTM(256)(se3)
        print(se4.shape)

        # decoder model
        decoder1 = add([fe2, se4])
        print(decoder1.shape)
        decoder2 = Dense(256, activation='relu')(decoder1)
        outputs = Dense(vocab_size, activation='softmax')(decoder2)

        model = Model(inputs=[inputs1, inputs2], outputs=outputs)
        model.compile(loss='categorical_crossentropy', optimizer='adam')

        # plot the model
        plot_model(model, show_shapes=True)
        return model

    @staticmethod
    def load_caption(captions_path):
        # Load Caption CSV file
        # os.path.join(working_dir,'drive/MyDrive/Collab_Projects/CaptionGenerator/flickr30k_images/results.csv')
        with open(captions_path, 'r') as f:
            next(f)  # Since the csv file has both image and caption,we just want captions
            captions = f.read()
        # Map imageId to captions
        caption_map = {}
        for line in tqdm(captions.split('\n')):
            token = line.split('|')
            if len(token) < 3:
                continue
            image_Id, comment_number, caption = token[0], token[1], token[2]
            # print(image_Id)
            # print(comment_number)
            # print(caption)
            # print(token)
            # break
            # Remove extension from image id
            image_Id = image_Id.split('.')[0]
            # print(image_Id)

            # print(caption)
            # Create list
            if image_Id not in caption_map:
                caption_map[image_Id] = []
                # store the caption
            caption_map[image_Id].append(caption)
        return caption_map

    # Load the Vgg16 model

    @staticmethod
    def get_tokenizer(caption_map, dir):
        captions_list = []
        for key in caption_map:
            for caption in caption_map[key]:
                captions_list.append(caption)
        # Tokenize the text
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(captions_list)  # Convert the words into tokens
        vocab_size = len(tokenizer.word_index) + 1  # tokenizer.word_index gives the dictory mapping each word to tokens
        print(vocab_size)
        with open(os.path.join(dir, 'tokenizer.pkl'), 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return tokenizer, captions_list, vocab_size

    @staticmethod
    def idx_to_word(integer, tokenizer):
        for word, index in tokenizer.word_index.items():
            if index == integer:
                return word
        return None

    @staticmethod
    def predict_caption(model, image, tokenizer, max_length):
        # add start tag for generation process
        in_text = 'startseq'
        # iterate over the max length of sequence
        for i in range(max_length):
            # encode input sequence
            sequence = tokenizer.texts_to_sequences([in_text])[0]
            # pad the sequence
            sequence = pad_sequences([sequence], max_length)
            # predict next word
            yhat = model.predict([image, sequence], verbose=0)
            # get index with high probability
            yhat = np.argmax(yhat)
            # convert index to word
            word = idx_to_word(yhat, tokenizer)
            # stop if word not found
            if word is None:
                break
            # append word as input for generating next word
            in_text += " " + word
            # stop if we reach end tag
            if word == 'endseq':
                break
        return in_text

    @staticmethod
    def generate_image_description():
        model_lstm = load_model('/model_checkpoint.h5')
        vgg_model = VGG16()
        # restructure the model
        vgg_model = Model(inputs=vgg_model.inputs,
                          outputs=vgg_model.layers[-2].output)
        image_id = '10002456'  # '1000092795'

        image_path = os.path.join(
            'drive/MyDrive/Collab_Projects/CaptionGenerator/flickr30k_images/flickr30k_images/' + image_id + '.jpg')
        # load image
        image = load_img(image_path, target_size=(224, 224))

        image_pil = Image.open(image_path)
        plt.imshow(image_pil)
        captions = ImageModule.caption_map[image_id]
        print('---------------------Actual---------------------')
        for caption in captions:
            print(caption)
        # convert image pixels to numpy array
        image = img_to_array(image)
        # reshape data for model
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        # preprocess image from vgg
        image = preprocess_input(image)
        # extract features
        feature = vgg_model.predict(image, verbose=0)  # np.expand_dims(features[image_id], axis=0)#
        # predict from the trained model
        y_pred = ImageModule.predict_caption(model_lstm, feature, tokenizer, max_length)
        print('--------------------Predicted--------------------')
        print(y_pred)

    @staticmethod
    def image_process_premodel():

        model_vgg = VGG16()
        print(type(model_vgg.inputs))
        print(type(model_vgg.output))
        # Restructure the model
        model_vgg = Model(inputs=model_vgg.inputs,
                          outputs=model_vgg.layers[-2].output)  # Not taking the vgg16 fully connected model

        # Summarize the model
        print(model_vgg.summary())

        base_dir = "/content/drive/MyDrive/Collab_Projects/CaptionGenerator"
        working_dir = '/content'
        ImageModule.save_features(base_dir)
        features = ImageModule.load_features(base_dir)
        caption_map = ImageModule.load_caption(captions_path=os.path.join(working_dir,
                                                                          'drive/MyDrive/Collab_Projects/CaptionGenerator/flickr30k_images/results.csv'))
        tokenizer, captions_list, vocab_size = ImageModule.get_tokenizer(caption_map, base_dir)

        # Get max length of caption
        max_length = max(len(caption.split()) for caption in captions_list)
        print(max_length)

        ##Train Test Split

        image_Ids = list(caption_map.keys())
        print(len(image_Ids))
        split = int(len(image_Ids) * 0.80)
        split1 = int(len(image_Ids) * 0.90)
        train = image_Ids[:split]
        test = image_Ids[split1:]
        val = image_Ids[split:split1]
        print(len(train))
        print(len(test))
        print(len(val))

        model_lstm = ImageModule.create_lstm_model(vocab_size, max_length)

        # Train the model
        epochs = 50
        batch_size = 64
        steps = len(train) // batch_size  # After each step it will do bakcpropagation and fetch the data

        generator = ImageModule.data_loader(train, caption_map, features, tokenizer, max_length, vocab_size, batch_size)
        val_generator = ImageModule.data_loader(val, caption_map, features, tokenizer, max_length, vocab_size,
                                                batch_size)

        # Define a learning rate schedule (example: reducing learning rate when a metric has stopped improving)
        lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5)

        checkpoint = ModelCheckpoint(base_dir + 'model_checkpoint.h5', save_best_only=True, monitor='val_loss',
                                     mode='min')
        early_stopping = EarlyStopping(monitor='val_loss', patience=10)
        val_steps = len(val) // batch_size
        model_lstm.fit(generator, epochs=epochs, steps_per_epoch=steps, verbose=1,
                       validation_data=val_generator,
                       validation_steps=val_steps,  # define this similarly to `steps`
                       callbacks=[checkpoint, early_stopping, lr_scheduler])
        ImageModule.generate_image_description()

    @staticmethod
    # Convert onehot vector into word
    def idx_to_word(integer, tokenizer):
        for word, index in tokenizer.word_index.items():
            if index == integer:
                return word
        return None

    # generate image description for an image
    @staticmethod
    def predict_caption(model, image, tokenizer, max_length):
        # add start tag for generation process
        in_text = 'startseq'
        # iterate over the max length of sequence
        for i in range(max_length):
            # encode input sequence
            sequence = tokenizer.texts_to_sequences([in_text])[0]
            # pad the sequence
            sequence = pad_sequences([sequence], max_length)
            # predict next word
            yhat = model.predict([image, sequence], verbose=0)
            # get index with high probability
            yhat = np.argmax(yhat)
            # convert index to word
            word = ImageModule.idx_to_word(yhat, tokenizer)
            # stop if word not found
            if word is None:
                break
            # append word as input for generating next word
            in_text += " " + word
            # stop if we reach end tag
            if word == 'endseq':
                break
        return in_text

    @staticmethod
    def image_description_generator(image_path, max_length=82):
        # from keras.models import load_model #to load saved lstm model
        model_lstm = load_model(
            '/Users/Manisha/Documents/MS/SDSU/course/BDA-696/final_project/project/ImageDrivenCaptioningAndCustomMusicRecommendations/resources/model/lstm_model.h5')
        # Testing with real Image
        vgg_model = VGG16()
        # restructure the model
        vgg_model = Model(inputs=vgg_model.inputs,
                          outputs=vgg_model.layers[-2].output)
        # load image
        image = load_img(image_path, target_size=(224, 224))
        # convert image pixels to numpy array
        image = img_to_array(image)
        # reshape data for model
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        # preprocess image from vgg
        image = preprocess_input(image)
        # extract features
        feature = vgg_model.predict(image, verbose=0)  # np.expand_dims(features[image_id], axis=0)#
        # load tokenizer from pickle file
        with open(
                '/Users/Manisha/Documents/MS/SDSU/course/BDA-696/final_project/project/ImageDrivenCaptioningAndCustomMusicRecommendations/resources/model/tokenizer2.pkl',
                'rb') as f:
            tokenizer = pickle.load(f)
        # predict from the trained model
        return ImageModule.predict_caption(model_lstm, feature, tokenizer, max_length)


def image_processing(image_path):
    max_length = 82
    # print(image_path)
    return ImageModule.image_description_generator(image_path, max_length=max_length)

# print(image_processing("/Users/Manisha/Documents/MS/SDSU/course/BDA-696/final_project/project/ImageDrivenCaptioningAndCustomMusicRecommendations/resources/datasets/sample_image.jpg"))
