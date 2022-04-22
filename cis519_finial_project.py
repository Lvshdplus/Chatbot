# -*- coding: utf-8 -*-
"""CIS519_Finial_Project.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1emz3Z9fw5gq00IWXcbs-Bxyaks_R8ZZq

# 1. Data Preprocess

## 1.1 Data Collection 

Download the datasets from [Cornell Movie Datasets Website](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html) and unzip the data into txt files.
"""

! wget -nc "http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip"
! unzip cornell_movie_dialogs_corpus.zip
! rm cornell_movie_dialogs_corpus.zip

"""## 1.2 Data Cleaning & Wrangling

Clean the data and convert it into the form of a dialog.
"""

# open dialog files
movie_lines = open('cornell movie-dialogs corpus/movie_lines.txt', encoding='utf-8',errors='ignore').read().split('\n')
movie_conversations = open('cornell movie-dialogs corpus/movie_conversations.txt', encoding='utf-8',errors='ignore').read().split('\n')

# build a dictionary to record (line_number, dialog) mappings
line_to_dialog = {}
for line in movie_lines:
  line_splited = line.split(' +++$+++ ')
  line_to_dialog[line_splited[0]] = line_splited[-1]

# build dialog fragments
dialog_fragments = []
for conversation in movie_conversations:
  ## convert u0 +++$+++ u2 +++$+++ m0 +++$+++ ['L194', 'L195', 'L196', 'L197'] to 'L194', 'L195', 'L196', 'L197'
  dialog_instance = conversation.split(' +++$+++ ')[-1][1:-1]
  ## convert 'L194', 'L195', 'L196', 'L197' to ['L194', 'L195', 'L196', 'L197']
  dialog_fragments.append(dialog_instance.replace("'", " ").replace(",","").split())

# convert dialog fragments into (question, answer) pairs
questions = []
answers = []

for frag in dialog_fragments:
    for i in range(1, len(frag)):
        questions.append(line_to_dialog[frag[i-1]])
        answers.append(line_to_dialog[frag[i]])

# show questions and answers pairs
def qa_show(num):
  for i in range(num):
    print('-------------------------------------------------\n')
    print(f"Dialog A: {questions[i]}\n")
    print(f"Dialog B: {answers[i]}\n")

qa_show(5)

"""## 1.3 Text Preprocessing

Preprocess the texts.

"""

# Global variables
TEXT_LIMIT = 15

# filter out long dialogs
def filter_long_texts(questions, answers, limit):
    short_questions = []
    short_answers = []
    for i in range(len(questions)):
        # if len(questions[i]) <= TEXT_LIMIT and len(answers[i]) <=TEXT_LIMIT:
        if len(questions[i].split()) <= TEXT_LIMIT:
            short_questions.append(questions[i])
            short_answers.append(answers[i])
    return short_questions, short_answers

filtered_questions, filtered_answers = filter_long_texts(questions, answers, TEXT_LIMIT)

# clean the texts
import re

replacement_patterns = [
  (r'won\'t', 'will not'),
  (r'can\'t', 'cannot'),
  (r'i\'m', 'i am'),
  (r'ain\'t', 'is not'),
  (r'(\w+)\'ll', '\g<1> will'),
  (r'(\w+)n\'t', '\g<1> not'),
  (r'(\w+)\'ve', '\g<1> have'),
  (r'(\w+)\'s', '\g<1> is'),
  (r'(\w+)\'re', '\g<1> are'),
  (r'(\w+)\'d', '\g<1> would'),
]

class TextCleaner(object):
  def __init__(self, patterns=replacement_patterns):
    self.patterns = [(re.compile(regex), repl) for (regex, repl) in patterns]
    
  def replace(self, text):
    s = text
    for (pattern, replace) in self.patterns:
      s = re.sub(pattern, replace, s)
    return s

cleaner = TextCleaner()

# Function for preprocessing the given text
def preprocess_text(text):
    
    # Lowercase the text
    text = text.lower()
    
    # Decontracting the text (e.g. it's -> it is)
    text = cleaner.replace(text)
    
    # Remove the punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)

    # Remove extra spaces
    text = re.sub(r"[ ]+", " ", text)
    
    return text

cleaned_questions = [preprocess_text(que) for que in filtered_questions]
cleaned_answers = [' '.join(ans.split()[:TEXT_LIMIT-2]) for ans in filtered_answers]

del(questions, answers, filtered_questions, filtered_answers)

for i in range(5):
    print('-------------------------------------------------\n')
    print(f'Dialog A: {cleaned_questions[i]}\n')
    print(f'Dialog B: {cleaned_answers[i]}\n')

"""After preprocessing the dataset, we should add a start tag (e.g. `<start>`) and an end tag (e.g. `<end>`) to answers. Remember that we will only add these tags to answers and not questions.

When Seq2Seq model is generating the word answers, we can first send it the `<start>` to begin the word generation. When `<end>` is generated, we will stop the iteration.

"""

cleaned_answers = ["starttoken " + ans + " endtoken" for ans in cleaned_answers]

# trim the data in case of running out of RAM
# TRAINING_SIZE = 170000
training_questions = cleaned_questions
training_answers = cleaned_answers
# training_questions = cleaned_questions[:TRAINING_SIZE]
# training_answers = cleaned_answers[:TRAINING_SIZE]

# testing_questions = cleaned_questions[TRAINING_SIZE:]
# testing_answers = cleaned_answers[TRAINING_SIZE:]

"""## 1.4 Input Encoding

Convet String input into numerical values

"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import preprocessing, utils
from tensorflow.keras.preprocessing.sequence import pad_sequences

NUM_WORDS = 8000

# Initialize the tokenizer
tokenizer = preprocessing.text.Tokenizer(num_words = NUM_WORDS, oov_token='nulltoken')

# Fit the tokenizer to questions and answers
tokenizer.fit_on_texts(training_questions + training_answers)

# Get the total vocab size
VOCAB_SIZE = len(tokenizer.word_index) + 1
# display(tokenizer.word_index)
print( 'VOCAB SIZE : {}'.format(VOCAB_SIZE))

tokenizer.word_index["starttoken"]

### encoder input data

# Tokenize the questions
tokenized_questions_training = tokenizer.texts_to_sequences(training_questions)
# tokenized_questions_testing = tokenizer.texts_to_sequences(testing_questions)

# Pad the sequences
padded_questions_training = pad_sequences(tokenized_questions_training, maxlen=TEXT_LIMIT, padding='post')
# padded_questions_testing = pad_sequences(tokenized_questions_testing, maxlen=TEXT_LIMIT, padding='post')

# Convert the sequences into array
encoder_input_data_training = np.array(padded_questions_training)
print(encoder_input_data_training.shape, TEXT_LIMIT)
# encoder_input_data_testing = np.array(padded_questions_testing)
# print(encoder_input_data_testing.shape, TEXT_LIMIT)

### decoder input data

# Tokenize the answers
tokenized_answers_training = tokenizer.texts_to_sequences(training_answers)
# tokenized_answers_testing = tokenizer.texts_to_sequences(testing_answers)

# Pad the sequences
padded_answers_training = pad_sequences(tokenized_answers_training, maxlen=TEXT_LIMIT, padding='post')
# padded_answers_testing = pad_sequences(tokenized_answers_testing, maxlen=TEXT_LIMIT, padding='post')

# Convert the sequences into array
decoder_input_data_training = np.array(padded_answers_training)
print(decoder_input_data_training.shape, TEXT_LIMIT)

# decoder_input_data_testing = np.array(padded_answers_testing)
# print(decoder_input_data_testing.shape, TEXT_LIMIT)

### decoder_output_data

# Iterate through index of tokenized answers
for i in range(len(tokenized_answers_training)):
    tokenized_answers_training[i] = tokenized_answers_training[i][1:]

# for i in range(len(tokenized_answers_testing)):
#     tokenized_answers_testing[i] = tokenized_answers_testing[i][1:]

# Pad the tokenized answers
padded_answers_training = pad_sequences(tokenized_answers_training, maxlen = TEXT_LIMIT, padding = 'post')
# padded_answers_testing = pad_sequences(tokenized_answers_testing, maxlen = TEXT_LIMIT, padding = 'post')

# One hot encode
onehot_answers_training = utils.to_categorical(padded_answers_training, NUM_WORDS+1)
# onehot_answers_testing = utils.to_categorical(padded_answers_testing, NUM_WORDS+1)

# Convert to numpy array
decoder_output_data_training = np.array(onehot_answers_training)
del(onehot_answers_training)

# decoder_output_data_testing = np.array(onehot_answers_testing)
# del (onehot_answers_testing)

print(decoder_output_data_training.shape)
# print(decoder_output_data_testing.shape)

# Import the libraries
import tensorflow.keras
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.activations import softmax
from tensorflow.keras.callbacks import ModelCheckpoint

print("Num GPUs Available: ", tf.config.experimental_list_devices())

# Hyper parameters
BATCH_SIZE = 32
EPOCHS = 120
VOCAB_SIZE = NUM_WORDS + 1

### Encoder Input
embed_dim = 50
num_lstm = 400

# Input for encoder
encoder_inputs = Input(shape = (None, ), name='encoder_inputs')

# Embedding layer
# Why mask_zero = True? https://www.tensorflow.org/guide/keras/masking_and_padding
encoder_embedding = Embedding(input_dim = VOCAB_SIZE, output_dim = embed_dim, mask_zero = True, name='encoder_embedding')(encoder_inputs)

# LSTM layer (that returns states in addition to output)
encoder_outputs, state_h, state_c = LSTM(units = num_lstm, return_state = True, name='encoder_lstm')(encoder_embedding)

# Get the states for encoder
encoder_states = [state_h, state_c]

### Decoder

# Input for decoder
decoder_inputs = Input(shape = (None,  ), name='decoder_inputs')

# Embedding layer
decoder_embedding = Embedding(input_dim = VOCAB_SIZE, output_dim = embed_dim , mask_zero = True, name='decoder_embedding')(decoder_inputs)

# LSTM layer (that returns states and sequences as well)
decoder_lstm = LSTM(units = num_lstm , return_state = True , return_sequences = True, name='decoder_lstm')

# Get the output of LSTM layer, using the initial states from the encoder
decoder_outputs, _, _ = decoder_lstm(inputs = decoder_embedding, initial_state = encoder_states)

# Dense layer
decoder_dense = Dense(units = VOCAB_SIZE, activation = softmax, name='decoder_outputs') 

# Get the output of Dense layer
output = decoder_dense(decoder_outputs)

# Create the model
model = Model([encoder_inputs, decoder_inputs], output)

# Compile the model
model.compile(optimizer='adam', metrics=['acc'], loss = "categorical_crossentropy")

# Summary
model.summary()

# Train the model
model.fit(
    x = {"encoder_inputs": encoder_input_data_training, "decoder_inputs": decoder_input_data_training}, 
    y = {'decoder_outputs': decoder_output_data_training}, 
    batch_size = BATCH_SIZE, 
    epochs = EPOCHS,
    shuffle = True,
    validation_split = 0.1
)

model.save(filepath=f"weight_{EPOCHS}.h5")

"""# Inference

Build the inference model(the same as training model) and load the trained weight. 
"""

# Load the final model
model.load_weights('weight_200.h5') 
print("Model Weight Loaded!")

# Function for making inference
def make_inference_models():
    
    # Create a model that takes encoder's input and outputs the states for encoder
    encoder_model = Model(inputs=encoder_inputs, outputs=encoder_states)
    
    # Create two inputs for decoder which are hidden state (or state h) and cell state (or state c)
    decoder_state_input_h = Input(shape = (num_lstm, ), name='decoder_state_input_h')
    decoder_state_input_c = Input(shape = (num_lstm, ), name='decoder_state_input_c')
    
    # Store the two inputs for decoder inside a list
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    
    # Pass the inputs through LSTM layer you have created before
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_embedding, initial_state = decoder_states_inputs)
    
    # Store the outputted hidden state and cell state from LSTM inside a list
    decoder_states = [state_h, state_c]

    # Pass the output from LSTM layer through the dense layer you have created before
    decoder_outputs = decoder_dense(decoder_outputs)

    # Create a model that takes decoder_inputs and decoder_states_inputs as inputs and outputs decoder_outputs and decoder_states
    decoder_model = Model([decoder_inputs] + decoder_states_inputs,
                          [decoder_outputs] + decoder_states)
    
    return encoder_model , decoder_model

# Function for converting strings to tokens
def str_to_tokens(sentence: str, tokenizer, maxlen_questions=TEXT_LIMIT):

    # Lowercase the sentence and split it into words
    words = sentence.lower().split()

    tokens_list = tokenizer.texts_to_sequences([sentence])

    # Pad the sequences to be the same length
    return pad_sequences(tokens_list , maxlen = maxlen_questions, padding = 'post')

# Initialize the model for inference
enc_model , dec_model = make_inference_models()

# Iterate through the number of times you want to ask question
try:
    for _ in range(5):

        # Get the input and predict it with the encoder model
        encoder_inputs = str_to_tokens(preprocess_text(input('Enter question : ')), tokenizer)
        states_values = enc_model.predict(encoder_inputs)

        # Initialize the decoder input sequence with the starttoken index
        # Reshape this to be a (1, 1) array, since it is a sequence of 1 sample for 1 timestep
        empty_target_seq = np.zeros((1, 1))

        # Update the target sequence with index of "start"
        empty_target_seq[0, 0] = tokenizer.word_index["starttoken"]
        # Initialize the stop condition with False
        stop_condition = False

        # Initialize the decoded words with an empty string
        decoded_translation = []

        # While stop_condition is false
        while not stop_condition :

            # Predict the (target sequence + the output from encoder model) with decoder model
            dec_outputs , h , c = dec_model.predict([empty_target_seq] + states_values)
            # Get the index for sampled word using the dec_outputs
            # dec_outputs is a numpy array of the shape (sample, timesteps, VOCAB_SIZE)
            # To start, we can just pick the word with the higest probability - greedy search
            sampled_word_index = np.argmax(dec_outputs[0, -1, :])
            
            # Initialize the sampled word with None
            sampled_word = None

            # Iterate through words and their indexes
            for word, index in tokenizer.word_index.items() :

                # If the index is equal to sampled word's index
                if sampled_word_index == index :

                    # Add the word to the decoded string
                    decoded_translation.append(word)

                    # Update the sampled word
                    sampled_word = word

            # If sampled word is equal to "end" OR the length of decoded string is more that what is allowed
            if sampled_word == 'endtoken' or len(decoded_translation) > TEXT_LIMIT:

                # Make the stop_condition to true
                stop_condition = True

            # Initialize back the target sequence to zero - array([[0.]])    
            empty_target_seq = np.zeros(shape = (1, 1))  

            # Update the target sequence with index of "start"
            empty_target_seq[0, 0] = sampled_word_index

            # Get the state values
            states_values = [h, c] 

            # Print the decoded string
        print(' '.join(decoded_translation[:-1]))
except KeyboardInterrupt:
    print('Ending conversational agent')
