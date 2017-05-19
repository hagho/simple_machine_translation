from __future__ import unicode_literals, print_function, division
from flask import Flask, request, render_template, jsonify


import unicodedata
from io import open
import random
import re

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()
torch.cuda.set_device(3)
SOS_token = 0
EOS_token = 1

MAX_LENGTH = 1000
class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        if self.name == "chn":
            for word in sentence:
                self.addWord(word)
            return
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    #s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    #s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    #pairs = [[s for s in l.split('\t')] for l in lines]
    
    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    #pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    #print(input_lang.name, input_lang.n_words, input_lang.word2count)
    #print(output_lang.name, output_lang.n_words, output_lang.word2count)
    return input_lang, output_lang, pairs


input_lang, output_lang, pairs = prepareData('eng', 'chn', True)
for i in random.choice(pairs):
    print(i)
print(random.choice(pairs))

def indexesFromSentence(lang, sentence):
    if lang.name == "chn":
        return [lang.word2index[word] for word in sentence]
    return [lang.word2index[word] for word in sentence.split(' ')]


def variableFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    result = Variable(torch.LongTensor(indexes).view(-1, 1))
    if use_cuda:
        return result.cuda()
    else:
        return result


def variablesFromPair(pair):
    input_variable = variableFromSentence(input_lang, pair[0])
    target_variable = variableFromSentence(output_lang, pair[1])
    return (input_variable, target_variable)


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        for i in range(self.n_layers):
            output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_output, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)))
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        for i in range(self.n_layers):
            output = F.relu(output)
            output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]))
        return output, hidden, attn_weights

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result

def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    input_variable = variableFromSentence(input_lang, sentence)
    input_length = input_variable.size()[0]
    encoder_hidden = encoder.initHidden()

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei],
                                                 encoder_hidden)
        encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))  # SOS
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    decoded_words = []
    decoder_attentions = torch.zeros(max_length, max_length)

    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_output, encoder_outputs)
        decoder_attentions[di] = decoder_attention.data
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(output_lang.index2word[ni])
        
        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    return decoded_words, decoder_attentions[:di + 1]

def evaluateSentence(encoder, decoder, sentence):
    
    print('>', sentence)

    output_words, attentions = evaluate(encoder, decoder, sentence)
    output_sentence = ' '.join(output_words)
    return output_sentence

encoder1 = torch.load("./encoder.pt")
attn_decoder1 = torch.load("./decoder.pt")

















app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/notebook')
def notebook():
    return render_template('notebook.html')



@app.route('/query', methods=['POST'])
def query():
    query_sen = request.form.get('query_sentence', '')
    '''
    k_input = int(request.form.get('k_input', ''))
    query_img = request.files['query_image']
    img_name = query_img.filename
    upload_img = os.path.join(app.config['UPLOAD_FOLDER'], img_name)
    sim_images, sim_image_degree = [], []
    '''
    sim_texts, sim_text_degree = [], []
    '''
    if img_name:
        query_img.save(upload_img)
        img_vec = image_transform(Image.open(upload_img).convert('RGB')).unsqueeze(0)
        image_emb = model.image_fc(resnet(Variable(img_vec))).squeeze(0)
        print 'image_emb mean: ', image_emb.mean()
        distance, indices = texts_nbrs.kneighbors(image_emb.data.numpy().reshape(1, -1))
        sim_text_degree = 1-distance[0][:k_input]/distance[0][-1]
        sim_texts = np.array(text_orig_all)[indices[0][:k_input]]
        sim_texts, sim_text_degree = sim_texts.tolist(), sim_text_degree.tolist()
    
    if query_sen:
        sentence = Variable(sentence_transform(query_sen)).unsqueeze(0)
        _, (sentence_emb, _) = model.rnn(sentence)
        # print 'sentence_emb mean: ', sentence_emb.mean()
        sentence_emb = sentence_emb.squeeze(0) * 1000
        sentence_emb = sentence_emb.squeeze(0)
        distance, indices = images_nbrs.kneighbors(sentence_emb.data.numpy().reshape(1, -1))

        sim_image_degree = 1-distance[0][:k_input]/distance[0][-1]
        sim_images = np.array(image_path_all)[indices[0][:k_input]]
        sim_images, sim_image_degree = sim_images.tolist(), sim_image_degree.tolist()

    upload_img = upload_img if img_name else 'no_upload_img'
    '''


    if query_sen:
        res = evaluateSentence(encoder1, attn_decoder1, query_sen)
        sim_texts.append(res)

    return jsonify(sim_texts=sim_texts)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=6789)
