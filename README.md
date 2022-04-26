## ChatChat – A Conversational Based Agent

**Team **: Shidong Lyu, Yao Shen, Yilin Guo. **Project Mentor TA **: Saumya


#### 1) Abstract

Build a Chatbot using LSTM layers and Seq2Seq model.


#### 2) Introduction

**Motivation:** 

We learned concepts about Natural Language Processing (NLP) from lectures and built up a deep averaging network in the homework. NLP has many applications, one of which is communication. There are many frontrunners in the personal assistant space: Siri, Alexa, Cortana, Google Assistant, etc. We are going to pursue this field by ourselves and develop a chatbot to simulate conversations with users in English. This can help users find “someone” to chat with when they are bored or feel lonely. What’s more, since we are training this chatbot with a dataset from movies, some responses would be dramatic and bring more happiness to users. 

**Problem Setup:** 

Our conversational based agent – ChatChat, is an end-to-end conversational agent.  

Inputs: A string of text representing a sentence. Words in the sentence are delimited with space.

Outputs: A string of text representing responses to the input sentence.

We will use conversations from movie scripts as our dataset. 

After extracting and preprocessing the conversations, we use 90% of dialogues to train our RNN model, and use the rest 10% dialogues to do tests.


#### 3) Background

In the famous Turing test, a machine is considered as ‘intelligent’ if it can exhibit behavior indistinguishable from that of a human (Turing, 1950). The chatbot is a real-world practice of this idea. The first chatbot ELIZA is an early natural language processing computer program created at MIT in 1966 (Weizenbaum, 1966). It uses pattern matching and substitution to give a response from a list of pre-programmed possible responses, so that it creates an illusion of understanding and interactions. ALICE, the first online chatbot developed by Richard Wallance in 1995, was inspired by ELIZA and based on pattern-matching, but utilized more natural language processing methodologies to realize more sophisticated conversations. Up till this point, chatbot was mainly powered by rule-based NLP technologies. When it moves to the new century and with the development of deep neural networks and representation learning, chatbots become more and more intelligent and reach people’s daily lives from customer queries to personal assistance. Apple Siri, Microsoft Cortana, Amazon Alexa etc. bring significant business profits and social benefits.

In the implementation of our ChatChat project, we mainly focus on adopting mature and extensive methodologies related to those introduced in class. The details of these related work are as follows:



1. S. Hochreiter and J. Schmidhuber, "Long Short-Term Memory," in _Neural Computation_, vol. 9, no. 8, pp. 1735-1780, 15 Nov. 1997, doi: 10.1162/neco.1997.9.8.1735. \
LSTM is the core technique in our project. Aside from the materials in lecture, we also refer to the original paper for more details. It is capable of handling long-term dependencies in sequence prediction predictions, which is essential in our task of speech recognition and response generation.
2. Cho, Kyunghyun, et al. "Learning phrase representations using RNN encoder-decoder for statistical machine translation." arXiv preprint arXiv:1406.1078 (2014). \
Link: [https://arxiv.org/abs/1406.1078](https://arxiv.org/abs/1406.1078) \
Our project is divided into two parts: understand and answer, which both deal with the sequence of linguistic phrases.  The RNN Encoder-Decoder model proposed by this article encodes a sequence of symbols to a fixed-length vector representation, and decodes the representation into another sequence of symbols, which are jointly trained together to maximize the conditional probability of a target sequence given a source sequence. This model is used in our project to give a semantically and syntactically meaningful response.
3. Tutorial of Learning phrase representations using RNN encoder-decoder for statistical machine translation \
Link: [https://github.com/bentrevett/pytorch-seq2seq/blob](https://github.com/bentrevett/pytorch-seq2seq/blob/master/2%20-%20Learning%20Phrase%20Representations%20using%20RNN%20Encoder-Decoder%20for%20Statistical%20Machine%20Translation.ipynb) \
There is a pytorch tutorial about the implementation of the above paper. It is not used in our project implemented with tensorflow, but it has detailed explanations and graphs, and is a good reference for us to understand how to implement the module of encoder and decoder.


#### 4) Summary of Our Contributions



1. **Contribution(s) in Application: **
    1. Build a chatbot that is capable of answering questions in terms of daily dialogues and some other interesting topics.
    2. Test different hyperparameters and different RNN structures to find the best model for our model.


#### 5) Detailed Description of Contributions

The entire project follows the procedure as below: (for more details, please refer to the Colab Notebook which includes all runnable codes and corresponding illustrations and explanations of why and how we did it.)



1. **Data Preprocess**:
    1. **Data Collection**: download / unzip datasets from [Cornell Movie Datasets](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html)
    2. **Data Cleaning & Wrangling**: Load the datasets and convert it into the form of dialogues(mainly focus on “movie_lines.txt” and “movie_conversations.txt”).
    3. **Text Preprocessing**:
        1. <span style="text-decoration:underline;">Filter out long Q&A</span>: long sequences are more likely to cause gradient vanishing problems, so we prefer to keep relatively short answers
        2. <span style="text-decoration:underline;">Process the texts</span>: It is necessary to remove punctuations and contractions in the sentences and make texts into lower cases.
        3. <span style="text-decoration:underline;">Add Tag Tokens</span>: add `starttoken` and `endtoken` to answers as markers for starting and ending an inference.
    4. **Text Encoding**:
        4. <span style="text-decoration:underline;">Choose Vocabulary Size</span>: Count words frequency and choose the most common portion of the dataset
        5. <span style="text-decoration:underline;">Build the Vocabulary</span>: Utilize tensorflow.keras.preprocessing.Tokenizer to build the vocabulary and only pick the common words and replace less frequent words with `nulltoken`. The vocabulary is a mapping which converts String inputs into numerical indexes. 
        6. <span style="text-decoration:underline;">Prepare Encoder and Decoder Inputs</span>:  Since String inputs can’t be directly fed into the network, we need to convert words in questions and answers to corresponding unique numerical indexes in vocabulary and pad zeros to the end of short sequences to make all inputs have the same length/dimensions.
        7. <span style="text-decoration:underline;">Prepare Output Labels</span>: The training can be considered as a classification task and we utilize `to_categorical` to convert class vectors(integers) to binary class matrix as the training labels.
2. **Training RNN Network**
    5.  **Build the training model**
        8. <span style="text-decoration:underline;">Build Encoder part</span>: Parse question input vectors into an Embedding layer, set the input size as (1 + Number of common words kept) where 1 is used for `nulltoken`. Feed the embedding into the LSTM model and keep track of output states and hidden states.
        9. <span style="text-decoration:underline;">Build Decoder part</span>: Parse answer input vectors into an Embedding layer similar to Encoder part, feed the embedding together with encoder output stats and hidden states into decoder LSTM layer. A dense layer is connected to the LSTM and `softmax` is used to do the final classification.
        10. <span style="text-decoration:underline;">Configure the Model</span>: Concatenate the Encoder and Decoder parts, set optimizer to be `adam` because adaptive learning rate optimizer is generally the best choice for LSTM model, we also tried RMSProp which has very similar training results.
        11. <span style="text-decoration:underline;">Checkpoints Callback</span>: Since we are training the model in AWS sagemaker(or Colab). setting the checkpoint is a good practice to save the model in case of network issues or accidental interrupts.
    6. **Training the LSTM Model**
        12. <span style="text-decoration:underline;">Fit the Model</span>: 1) Feed in preprocessed data, 2)set defined hyperparameters, 3)shuffle the datasets for each epoch such that each batch of data can better represent the entire dataset and 4) split the input data into 90% training data and 10% validation data.
        13. <span style="text-decoration:underline;">Save the Model</span>: Save the model and weights with specific parameter notations. 
3. **Inference**
    7. **Build inference model and load weights**: the inference model is the same as training model
    8. **Preprocess the inference inputs**: Since the input type is slightly different from the training data, we have to apply the same Text preprocess techniques but in a slightly different way. 
    9. **Main Inference:** 1) read in user-entered questions and preprocess the data into the same form of training data 2) initialize an empty sequence with `starttoken` 3) keep inference until encounter a `endtoken` or sequence is too long 4)convert the predicted sequence back to words by using Vocabulary built before


##### 5.1 Methods

Recurrent Neural Networks (RNNs) [Rumelhart, 1985] are a class of neural networks that operates on a sequence of variables, consisting of optional hidden states and outputs while allowing previous outputs to be used as inputs. To adjust the model weights for improvements, we can train the model using a backpropagation algorithm such that the prediction gets closer and closer to the desired target. However, size of gradients matters because in simple RNN cells, all the historical information must be retained within hidden layer nodes using weights and it cannot remember information for very long. We can meet vanishing gradient problems if the gradient is too small or the explosion problem is too large. To address the vanishing gradient problem, Long Short Term Memory Cell is used with a separate cell state

 <img src="https://lh5.googleusercontent.com/d_kMX1ODXfQaQpemdS1hrWO1xXzCPfflMzEfv9PHAycqWinty5kEOX6OIfFc4Tsor_m26U9LUdUVq00fQe257zmzkaLCp0rxC9rKBYFcmou0MwPFoRgVQvtx5mAbH3X8Iw88lBXC" alt="img" style="zoom:33%;" /><img src="https://lh4.googleusercontent.com/-tFOcnUZuE4gH4TmXTi-mJBrtqysIBFgYanS0JSnitEHQ59RHj38hU1qSotiG99F_GawRsJkVjNPVB19EHn7gycBBRXJwuUo54i1zpJuhraE0z-WmlCiNXExdR5IIlNW5RyAsyZi" alt="img" style="zoom: 50%;" />
 to remember historical information in a better way.  As shown in the following figure, the update gate can be used for adding information to the cell state, the forget gate can be used for removing information from the cell state, and the output gate is used for inputting to the hidden state. 

With LSTM as the fundamental component, we build our Seq2Seq model for our project. The Seq2Seq model consists of the Encoder RNN and Decoder RNN. The Encoder is a stack of several LSTM units which collect the information for each source word, propagate forward, and provide the initial hidden state for the Decoder RNN. The Decoder is composed of several LSTM units where each accept a hidden state from the previous unit (Encoder acts as the initial one), and predict the character of the target sequence at time _t_ as well as its hidden state for the next prediction. The major downside of this structure is that the Encoder captures lots of information about the source and crams to the initial hidden state. Thus during decoding, the hidden state needs to contain all the information of the source sequence and decoded tokens, which leads to a bottleneck. To address this problem, we can adopt the technique of attention, where we use the attention distribution to take a weighted sum of the encoder hidden states, so that attention output can mostly contain information of the hidden states with high atten. Then concatenate attention output with decoder hidden state to use the normal Seq2Seq model. Another approach is to use the GRU (Gated Recurrent Unit) to replace LSTM. GRU [Cho, 2014] simplifies the LSTM while maintaining its ability to remember long term dependencies. Its performance is similar to LSTM but trains much faster and requires less memory.


##### 5.2 Experiments and Results

**1 Experiments on Vocabulary Size**

The entire dataset includes more than 180k+ questions and answers and 40k+ different words. Some of the words are not even English words, they may be Italian or other languages. If we take all words into consideration, then the input would be very large and too many less frequent words isn’t beneficial to our model training. 

![img](https://lh3.googleusercontent.com/haehOAOGkVBvrJmrnqWNH59Y29Y73HufwGYcP52S6I_DT-N-WF13y3evRc0rHSB15jSNMpbGFveRNDNu74CS77LsMuzZGn9aPYozCBKm6Re-NsiEaxWGe2W3Z6Q7z3enwIW0MUMs)

**2 Experiments on Sequence Length Limit**

Since RNN neural networks have gradient vanishing problems, we don’t want the sequence to be very long. Also the movie conversations include some descriptive and narrative sentences, short lines typically mean daily and normal dialogues. Filtering out long sentences can make the chatbot capable of answering questions in a more general way. 

**3 Experiments on different structures of LSTM Model**

Though the LSTM Model is great for NLP training, we have tried several embedding dimensions and number of units used in LSTM layers to see which is the best for our datasets. Besides, one layer LSTM is usually good for simple datasets, so we tried to stack multiple LSTM layers together to figure out the best structure for our datasets.

The following table shows the results of our experiments:

L: number of layers, D: embedding dimensions, U: number of units in LSTM


<table>
  <tr>
   <td>LSTM Structure 
   </td>
   <td>Q: only limit question Size
<p>
Q&A: limit both que/ans size
   </td>
   <td>Vocabulary Size
   </td>
   <td>Data Size
   </td>
   <td>Accuracy(training until acc not improving)
   </td>
  </tr>
  <tr>
   <td>1L, 32D, 128U 
   </td>
   <td>Q&A
   </td>
   <td>3500
   </td>
   <td>180k
   </td>
   <td>84%
   </td>
  </tr>
  <tr>
   <td>1L, 64D, 256U
   </td>
   <td>Q
   </td>
   <td>3500
   </td>
   <td>30k
   </td>
   <td>88%
   </td>
  </tr>
  <tr>
   <td>1L, 64D, 512U
   </td>
   <td>Q
   </td>
   <td>3500
   </td>
   <td>180k
   </td>
   <td>48%
   </td>
  </tr>
  <tr>
   <td>1L, 64D, 512U
   </td>
   <td>Q
   </td>
   <td>5000
   </td>
   <td>180k
   </td>
   <td>49%
   </td>
  </tr>
  <tr>
   <td>1L, 64D, 512U
   </td>
   <td>Q
   </td>
   <td>8000
   </td>
   <td>180k
   </td>
   <td>49%
   </td>
  </tr>
  <tr>
   <td>2L, 32D, 128U
   </td>
   <td>Q&A
   </td>
   <td>3500
   </td>
   <td>180k
   </td>
   <td>84%
   </td>
  </tr>
</table>


##### 5.3 Evaluations


###### 5.3.1 Performance

Based on the experiments above, the better choice of LSTM structure is 1 layer LSTM with 64 embedding dimensions and 256 units in the LSTM layer. The best accuracy converges when it closes to 90%, which means we still have to try other options both in width and depth of the RNN structure. The validation accuracy ranges from 35% ~ 50% for any structure and any number of epochs. The problem may be that the evaluation mechanism is not suitable for our model because the predicted results may have very similar meaning to desired answers but the words used may not be exactly the same as the words used in testing data. 


###### 5.3.2 Chatbot Conversation Snapshot


##### The chatbot can answer some general questions, but when you ask her more complex questions, it will give you `unexpected` answers. The `unexpected` here means that answers do make sense but will include some extra background information probably because the dialogues from our datasets are only pieces of entire conversations. The datasets themselves are not general questions and answers but are biased to some extent, including slang, causal expressions, jokes, etc.  


##### 5.4 Next Steps To do

We spent the whole last 2 weeks training different models. Since the data is very large and training RNN is very slow, we used up all AWS credits, but we still need to train more models with different structures and data preprocessing techniques, including 1) try multiple LSTM layers(more than 2), 2) using GRU layers instead of LSTM, 3) find a datasets with more general conversations(more formal) 4) remove non-English words. 


#### 6) Compute/Other Resources Used

1. Colab Pro, 2) Amazon SageMaker: 1 [ml.g4dn.2xlarge], 2 [ml.g4dn.16xlarg]


#### 7) Conclusions

We successfully trained a Chatbot capable of answering some common questions. During the tough process, we learned how to design a neural network, tune the hyperparameters and preprocess the data. We still have many things to try and test if time and resources permit.  
