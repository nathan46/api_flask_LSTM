
import numpy as np
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import spacy
from flask import Flask, jsonify, request
import codecs
import json
import torch.nn as nn
import torch
app = Flask(__name__)
import fr_core_news_sm
nlp = fr_core_news_sm.load()

seq_length=30

punctuation = "!\"#$%&'()*+-./:;<=>?@[\]^_`{|}~,"

with open('vocab_to_int-30k_3b.json') as f:
    vocab_to_int = json.load(f)

train_on_gpu= False#torch.cuda.is_available()

class SentimentLSTM(nn.Module):
    """
    The RNN model that will be used to perform Sentiment analysis.
    """

    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):
        """
        Initialize the model by setting up the layers.
        """
        super().__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        # embedding and LSTM layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers,
                            dropout=drop_prob, batch_first=True)

        # dropout layer
        self.dropout = nn.Dropout(0.3)

        # linear and sigmoid layers
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sig = nn.Sigmoid()


    def forward(self, x, hidden):
        """
        Perform a forward pass of our model on some input and hidden state.
        """
        batch_size = x.size(0)

        # embeddings and lstm_out
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)

        # stack up lstm outputs
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        # dropout and fully-connected layer
        out = self.dropout(lstm_out)
        out = self.fc(out)
        # sigmoid function
        sig_out = self.sig(out)

        # reshape to be batch_size first
        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1] # get last batch of labels

        # return last sigmoid output and hidden state
        return sig_out, hidden


    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        if (train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())

        return hidden

def pad_features(tweet_int, seq_length):
    ''' Return features of tweets_ints, where each tweet is padded with 0's or truncated to the input seq_length.
    '''
    features = np.zeros((len(tweet_int), seq_length), dtype = int)

    for i, tweet in enumerate(tweet_int):
        tweet_len = len(tweet)

        if tweet_len <= seq_length:
            zeroes = list(np.zeros(seq_length-tweet_len))
            new = zeroes+tweet

        elif tweet_len > seq_length:
            new = tweet[0:seq_length]

        features[i,:] = np.array(new)

    return features

def loadModel(name):
    vocab_size = len(vocab_to_int)+1 # +1 for the 0 padding
    output_size = 1
    embedding_dim = 200
    hidden_dim = 256
    n_layers = 2

    model = SentimentLSTM(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)
    model.load_state_dict(torch.load(name+'.pth', map_location='cpu'))
    model.eval()

    if(train_on_gpu):
        model.cuda()

    return model

net = loadModel('model_30k_3b')

def preprocess(tweet, vocab_to_int):
    tweet = tweet.lower()

    #enlever ponctuation
    tweet = ''.join([c for c in tweet if c not in punctuation])

    #enlever les liens
    indice = tweet.find("http")
    if indice == -1 :
        indice = tweet.find("pic.twitter")

    if indice != -1:
        tweet = tweet[:indice]

    #remove stopwords
    x = tweet.split()
    text_final = ""

    for word in x:
        if word not in stopwords.words('french'):
            text_final = text_final+" "+word

    tweet = text_final.strip()


    #lemmatisation
    doc = nlp(tweet)
    text_final = ""
    for token in doc:
        text_final = text_final+" "+token.lemma_

    tweet = text_final.strip()
    print(tweet)
    word_list = tweet.split()
    num_list = []
    #list of reviews
    #though it contains only one review as of now
    tweets_int = []
    for word in word_list:
        if word in vocab_to_int.keys():
            num_list.append(vocab_to_int[word])
    tweets_int.append(num_list)
    return tweets_int

def predict(net, test_tweet, sequence_length=30):
    ''' Prints out whether a give review is predicted to be
        positive or negative in sentiment, using a trained model.

        params:
        net - A trained net
        test_review - a review made of normal text and punctuation
        sequence_length - the padded length of a review
        '''
    #change the reviews to sequence of integers
    int_rev = preprocess(test_tweet, vocab_to_int)
    #pad the reviews as per the sequence length of the feature
    features = pad_features(int_rev, seq_length=seq_length)

    #changing the features to PyTorch tensor
    features = torch.from_numpy(features)

    #pass the features to the model to get prediction
    net.eval()
    val_h = net.init_hidden(1)
    val_h = tuple([each.data for each in val_h])

    if(train_on_gpu):
        features = features.cuda()

    output, val_h = net(features, val_h)

    #rounding the output to nearest 0 or 1
    #pred = torch.round(output)

    #mapping the numeric values to postive or negative
    #output = ["Positive" if pred.item() == 0 else "Negative"]

    # print custom response based on whether test_review is pos/neg
    return output.item()


@app.route("/", methods=['GET', 'POST'])
def index():
    if (request.method == 'POST'):
        some_json = request.get_json()
        tweet = some_json['tweet']
        pred = str(predict(net, tweet, seq_length))
        return "{'score':"+pred+"}", 201, {'Content-Type':'application/json'}
        #return jsonify({'score': pred}), 201
    else:
        return jsonify({"about": "Hello World!"})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=80)
