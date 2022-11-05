#!/usr/bin/env python
# coding: utf-8

# # CS 447 Homework 3 $-$ Neural Machine Translation
# In this homework we are going to perform machine translation using two deep learning approaches: a Recurrent Neural Network (RNN) and Transformer.
# 
# Specifically, we are going to train sequence to sequence models for Spanish to English translation. In this assignment you only need to implement the neural network models, we implement all the data loading for you. Please **refer** to the following resources for more details:
# 
# 1.   https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf
# 2.   https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
# 3. https://arxiv.org/pdf/1409.0473.pdf
# 
# <font color='green'>While you work, we suggest that you keep your hardware accelerator set to "CPU" (the default for Colab). However, when you have finished debugging and are ready to train your models, you should select "GPU" as your runtime type. This will speed up the training of your models. You can find this by going to <TT>Runtime > Change Runtime Type</TT> and select "GPU" from the dropdown menu.</font>
# 
# As usual, you should not import any other libraries.
# 

# # Step 1: Download & Prepare the Data

# In[1]:


### DO NOT EDIT ###

import pandas as pd
import unicodedata
import re
from torch.utils.data import Dataset
import torch
import random
import os
rnn_encoder, rnn_encoder, transformer_encoder, transformer_decoder = None, None, None, None


# ## Helper Functions
# This cell contains helper functions for the dataloader.

# In[2]:


### DO NOT EDIT ###

# Converts the unicode file to ascii
def unicode_to_ascii(s):
    """Normalizes latin chars with accent to their canonical decomposition"""
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')


def preprocess_sentence(w):
    '''
    Preprocess the sentence to add the start, end tokens and make them lower-case
    '''
    w = unicode_to_ascii(w.lower().strip())
    w = re.sub(r'([?.!,¿])', r' \1 ', w)
    w = re.sub(r'[" "]+', ' ', w)

    w = re.sub(r'[^a-zA-Z?.!,¿]+', ' ', w)
    
    w = w.rstrip().strip()
    w = '<start> ' + w + ' <end>'
    return w


def pad_sequences(x, max_len):
    padded = np.zeros((max_len), dtype=np.int64)
    if len(x) > max_len:
        padded[:] = x[:max_len]
    else:
        padded[:len(x)] = x
    return padded


def preprocess_data_to_tensor(dataframe, src_vocab, trg_vocab):
    # Vectorize the input and target languages
    src_tensor = [[src_vocab.word2idx[s if s in src_vocab.vocab else '<unk>'] for s in es.split(' ')] for es in dataframe['es'].values.tolist()]
    trg_tensor = [[trg_vocab.word2idx[s if s in trg_vocab.vocab else '<unk>'] for s in eng.split(' ')] for eng in dataframe['eng'].values.tolist()]

    # Calculate the max_length of input and output tensor for padding
    max_length_src, max_length_trg = max(len(t) for t in src_tensor), max(len(t) for t in trg_tensor)
    print('max_length_src: {}, max_length_trg: {}'.format(max_length_src, max_length_trg))

    # Pad all the sentences in the dataset with the max_length
    src_tensor = [pad_sequences(x, max_length_src) for x in src_tensor]
    trg_tensor = [pad_sequences(x, max_length_trg) for x in trg_tensor]

    return src_tensor, trg_tensor, max_length_src, max_length_trg


def train_test_split(src_tensor, trg_tensor):
    '''
    Create training and test sets.
    '''
    total_num_examples = len(src_tensor) - int(0.2*len(src_tensor))
    src_tensor_train, src_tensor_test = src_tensor[:int(0.75*total_num_examples)], src_tensor[int(0.75*total_num_examples):total_num_examples]
    trg_tensor_train, trg_tensor_test = trg_tensor[:int(0.75*total_num_examples)], trg_tensor[int(0.75*total_num_examples):total_num_examples]

    return src_tensor_train, src_tensor_test, trg_tensor_train, trg_tensor_test


# ## Download and Visualize the Data
# 
# Here we will download the translation data. We will learn a model to translate Spanish to English.

# In[3]:


### DO NOT EDIT ###

if __name__ == '__main__':
    os.system("wget http://www.manythings.org/anki/spa-eng.zip")
    os.system("unzip -o spa-eng.zip")


# Now we visualize the data.

# In[4]:


### DO NOT EDIT ###

if __name__ == '__main__':
    lines = open('spa.txt', encoding='UTF-8').read().strip().split('\n')
    total_num_examples = 50000 
    original_word_pairs = [[w for w in l.split('\t')][:2] for l in lines[:total_num_examples]]
    random.shuffle(original_word_pairs)

    dat = pd.DataFrame(original_word_pairs, columns=['eng', 'es'])
    print(dat) # Visualize the data


# Next we preprocess the data.

# In[5]:


### DO NOT EDIT ###

if __name__ == '__main__':
    data = dat.copy()
    data['eng'] = dat.eng.apply(lambda w: preprocess_sentence(w))
    data['es'] = dat.es.apply(lambda w: preprocess_sentence(w))
    print(data) # Visualizing the data


# ## Vocabulary & Dataloader Classes
# 
# First we create a class for managing our vocabulary as we did in Homework 2. In this homework, we have a separate class for the vocabulary as we need 2 different vocabularies $-$ one for English and one for Spanish.
# 
# Then we prepare the dataloader and make sure it returns the source sentence and target sentence.

# In[6]:


### DO NOT EDIT ###

class Vocab_Lang():
    def __init__(self, vocab):
        self.word2idx = {'<pad>': 0, '<unk>': 1}
        self.idx2word = {0: '<pad>', 1: '<unk>'}
        self.vocab = vocab
        
        for index, word in enumerate(vocab):
            self.word2idx[word] = index + 2 # +2 because of <pad> and <unk> token
            self.idx2word[index + 2] = word
    
    def __len__(self):
        return len(self.word2idx)

class MyData(Dataset):
    def __init__(self, X, y):
        self.length = torch.LongTensor([np.sum(1 - np.equal(x, 0)) for x in X])
        self.data = torch.LongTensor(X)
        self.target = torch.LongTensor(y)
    
    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        return x, y

    def __len__(self):
        return len(self.data)


# In[7]:


### DO NOT EDIT ###

import numpy as np
import random
from torch.utils.data import DataLoader


# In[8]:


### DO NOT EDIT ###

if __name__ == '__main__':
    # HYPERPARAMETERS (You may change these if you want, though you shouldn't need to)
    BATCH_SIZE = 64
    EMBEDDING_DIM = 256


# ## Build Vocabulary

# In[9]:


### DO NOT EDIT ###

def build_vocabulary(pd_dataframe):
    sentences = [sen.split() for sen in pd_dataframe]
    vocab = {}
    for sen in sentences:
        for word in sen:
            if word not in vocab:
                vocab[word] = 1
    return list(vocab.keys())

if __name__ == '__main__':
    src_vocab_list = build_vocabulary(data['es'])
    trg_vocab_list = build_vocabulary(data['eng'])


# ## Instantiate Datasets
# 
# We instantiate our training and validation datasets.

# In[10]:


### DO NOT EDIT ###

if __name__ == '__main__':
    src_vocab = Vocab_Lang(src_vocab_list)
    print(f"src_vocab is: {src_vocab}")
    trg_vocab = Vocab_Lang(trg_vocab_list)
    print(f"trg_vocab is: {trg_vocab}")

    src_tensor, trg_tensor, max_length_src, max_length_trg = preprocess_data_to_tensor(data, src_vocab, trg_vocab)
    src_tensor_train, src_tensor_val, trg_tensor_train, trg_tensor_val = train_test_split(src_tensor, trg_tensor)

    # Create train and val datasets
    train_dataset = MyData(src_tensor_train, trg_tensor_train)
    train_dataset = DataLoader(train_dataset, batch_size=BATCH_SIZE, drop_last=True, shuffle=True)

    test_dataset = MyData(src_tensor_val, trg_tensor_val)
    test_dataset = DataLoader(test_dataset, batch_size=BATCH_SIZE, drop_last=True, shuffle=False)


# In[11]:


### DO NOT EDIT ###

if __name__ == '__main__':
    idxes = random.choices(range(len(train_dataset.dataset)), k=5)
    src, trg =  train_dataset.dataset[idxes]
    print('Source:', src)
    print('Source Dimensions: ', src.size())
    print('Target:', trg)
    print('Target Dimensions: ', trg.size())


# # Step 2: Train a Recurrent Neural Network (RNN) [45 points]
# 
# Here you will write a recurrent model for machine translation, and then train and evaluate its results.
# 
# Here are some links that you may find helpful:
# 1. Attention paper: https://arxiv.org/pdf/1409.0473.pdf
# 2. Explanation of LSTM's & GRU's: https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21
# 3. Attention explanation: https://towardsdatascience.com/attention-in-neural-networks-e66920838742 
# 4. Another attention explanation: https://towardsdatascience.com/attention-and-its-different-forms-7fc3674d14dc
# 

# In[12]:


### DO NOT EDIT ###

import torch.nn as nn
import torch.nn.functional as F
import time
from tqdm.notebook import tqdm
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction, corpus_bleu


# ## <font color='red'>TODO:</font> Encoder Model [10 points]
# 
# First we build a recurrent encoder model, which will be very similar to what you did in Homework 2. However, instead of using a fully connected layer as the output, you should the return a sequence of outputs of your GRU as well as the final hidden state. These will be used in the decoder.
# 
# In this cell, you should implement the `__init(...)` and `forward(...)` functions, each of which is <b>5 points</b>.

# In[13]:


from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class RnnEncoder(nn.Module):
    def __init__(self, src_vocab, embedding_dim, hidden_units):
        super(RnnEncoder, self).__init__()
        """
        Args:
            src_vocab: Vocab_Lang, the source vocabulary
            embedding_dim: the dimension of the embedding
            hidden_units: The number of features in the GRU hidden state
        """
        self.src_vocab = src_vocab # Do not change
        vocab_size = len(src_vocab)

        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hidden_units = hidden_units

        ### TODO ###

        # Initialize embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embed_size = embedding_dim
        self.num_layers = 1


        # Initialize a single directional GRU with 1 layer and batch_first=False
        # self.GRU = nn.GRU(input_size=embedding_dim, hidden_size=hidden_units, num_layers=self.num_layers, batch_first=False).to(self.device)
        self.GRU = nn.GRU(input_size=embedding_dim, hidden_size=hidden_units, num_layers=self.num_layers, batch_first=False).to(device)
        # self.GRU = nn.GRU(input_size=embedding_dim, hidden_size=hidden_units, num_layers=self.num_layers, batch_first=False)

    def forward(self, x):
        """
        Args:
            x: source texts, [max_len, batch_size]

        Returns:
            output: [max_len, batch_size, hidden_units]
            hidden_state: [1, batch_size, hidden_units] 
        
        Pseudo-code:
        - Pass x through an embedding layer and pass the results through the recurrent net
        - Return output and hidden states from the recurrent net
        """
        output, hidden_state = None, None

        # TODO: We only have indexes of source text? and the size is [max_len, batch size]

        # print('Content of x:', x)
        # print('Shape of x:', x.shape, '\n')
        # print('Type of x:', x.dtype, '\n')

        ### TODO ###
        # batch_first – If True, then the input and output tensors are provided as (batch, seq, feature)
        #  Default: False, then the input and output tensors are provided as (seq, batch, feature).


        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Pass texts through your embedding layer to convert to word embeddings
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x_int64 = x.type(torch.int64)
        # print('Content of embedding:', x_int64)
        # print('Shape of embedding:', x_int64.shape, '\n')
        # print('Type of embedding:', x_int64.dtype, '\n')

        # Resulting: shape: [batch_size, max_len, embed_size]
        final_embedding = (self.embedding(x_int64))
        # final_embedding_gpu = final_embedding
        final_embedding_gpu = final_embedding.to(device)
        # print(f"  final_embedding_gpu: {final_embedding_gpu.shape}, final_embedding_gpu dtype: {final_embedding_gpu.dtype}, final_embedding_gpu device: {final_embedding_gpu.get_device()}")

        # print('Content of final_embedding_gpu:', final_embedding_gpu)

        max_len, batch_size = x_int64.shape
        # print(f"batch_size: {batch_size}, max_len: {max_len}")
        # print(f"(batch_size, max_len, self.hidden_size): ({batch_size}, {max_len}, {self.hidden_units})")

        initial_state_h0 = torch.nn.parameter.Parameter(torch.randn(1*self.num_layers, batch_size, self.hidden_units)).to(device)
        # initial_state_h0 = torch.nn.parameter.Parameter(torch.randn(1*self.num_layers, batch_size, self.hidden_units))
        # print(f"  initial_state_h0: {initial_state_h0.shape}, initial_state_h0 dtype: {initial_state_h0.dtype}, initial_state_h0 device: {initial_state_h0.get_device()}")

        # gru_input = torch.randn(batch_size, max_len, self.hidden_size).to(device)
        # print(f"  gru_input: {gru_input.shape}, gru_input dtype: {gru_input.dtype}, gru_input device: {gru_input.get_device()}")
        # # h_out = 32

        # Pass the result through your recurrent network
        #   See PyTorch documentation for resulting shape for nn.GRU
        output, hidden_state = self.GRU(final_embedding_gpu, initial_state_h0)
        # print(f"  encoder output shape: {output.shape}, encoder output dtype: {output.dtype}")
        # print(f" hidden_state_n: {hidden_state.shape}, hidden_state_n dtype: {hidden_state.dtype}")
        
        return output, hidden_state


# ## Sanity Check: RNN Encoder Model
# 
# The code below runs a sanity check for your `RnnEncoder` class. The tests are similar to the hidden ones in Gradescope. However, note that passing the sanity check does <b>not</b> guarantee that you will pass the autograder; it is intended to help you debug.

# In[14]:


### DO NOT EDIT ###

count_parameters = lambda model: sum(p.numel() for p in model.parameters() if p.requires_grad)

def sanityCheckModel(all_test_params, NN, expected_outputs, init_or_forward, data_loader):
    print('--- TEST: ' + ('Number of Model Parameters (tests __init__(...))' if init_or_forward=='init' else 'Output shape of forward(...)') + ' ---')
    if init_or_forward == "forward":
        # Reading the first batch of data for testing
        for texts_, labels_ in data_loader:
            texts_batch, labels_batch = texts_, labels_
            break

    for tp_idx, (test_params, expected_output) in enumerate(zip(all_test_params, expected_outputs)):       
        if init_or_forward == "forward":
            batch_size = test_params['batch_size']
            texts = texts_batch[:batch_size]
            if NN.__name__ == "RnnEncoder":
                texts = texts.transpose(0,1)

        # Construct the student model
        tps = {k:v for k, v in test_params.items() if k != 'batch_size'}
        stu_nn = NN(**tps)

        input_rep = str({k:v for k,v in tps.items()})

        if init_or_forward == "forward":
            with torch.no_grad():
                if NN.__name__ == "TransformerEncoder":
                    stu_out = stu_nn(texts)
                else:
                    stu_out, _ = stu_nn(texts)
                    expected_output = torch.rand(expected_output).transpose(0, 1).size()
            ref_out_shape = expected_output

            has_passed = torch.is_tensor(stu_out)
            if not has_passed: msg = 'Output must be a torch.Tensor; received ' + str(type(stu_out))
            else: 
                has_passed = stu_out.shape == ref_out_shape
                msg = 'Your Output Shape: ' + str(stu_out.shape)
            

            status = 'PASSED' if has_passed else 'FAILED'
            message = '\t' + status + "\t Init Input: " + input_rep + '\tForward Input Shape: ' + str(texts.shape) + '\tExpected Output Shape: ' + str(ref_out_shape) + '\t' + msg
            print(message)
        else:
            stu_num_params = count_parameters(stu_nn)
            ref_num_params = expected_output
            comparison_result = (stu_num_params == ref_num_params)

            status = 'PASSED' if comparison_result else 'FAILED'
            message = '\t' + status + "\tInput: " + input_rep + ('\tExpected Num. Params: ' + str(ref_num_params) + '\tYour Num. Params: '+ str(stu_num_params))
            print(message)

        del stu_nn


# In[15]:


### DO NOT EDIT ###

if __name__ == '__main__':
    # Set random seed
    torch.manual_seed(42)
    # Create test inputs
    embedding_dim = [2, 5, 8]
    hidden_units = [50, 100, 200]
    params = []
    inputs = []
    for i in range(len(embedding_dim)):
        for hu in hidden_units:
            inp = {}
            inp['src_vocab'] = src_vocab
            inp['embedding_dim'] = embedding_dim[i]
            inp['hidden_units'] = hu
            inputs.append(inp)

    # print(f"inputs are: {inputs}")

    # Test init
    expected_outputs = [33770, 56870, 148070, 72725, 96275, 188375, 111680, 135680, 228680]

    sanityCheckModel(inputs, RnnEncoder, expected_outputs, "init", None)
    print()

    # Test forward
    inputs = []
    batch_sizes = [1, 2]
    for hu in hidden_units:
        for b in batch_sizes:
            inp = {}
            inp['embedding_dim'] = EMBEDDING_DIM
            inp['src_vocab'] = src_vocab
            inp["batch_size"] = b
            inp['hidden_units'] = hu
            inputs.append(inp)
    # create sanity datasets
    sanity_dataset = MyData(src_tensor_train, trg_tensor_train)
    sanity_loader = torch.utils.data.DataLoader(sanity_dataset, batch_size=50, num_workers=2, drop_last=True, shuffle=True)
    expected_outputs = [torch.Size([1, 16, 50]), torch.Size([2, 16, 50]), torch.Size([1, 16, 100]), torch.Size([2, 16, 100]), torch.Size([1, 16, 200]), torch.Size([2, 16, 200])]

    sanityCheckModel(inputs, RnnEncoder, expected_outputs, "forward", sanity_loader)


# ## <font color='red'>TODO:</font> Decoder Model [15 points]
# We will implement a Decoder model that uses an attention mechanism, as provided in https://arxiv.org/pdf/1409.0473.pdf. We have broken this up into three functions that you need to implement: `__init__(self, ...)`, `compute_attention(self, dec_hs, enc_output)`, and `forward(self, x, dec_hs, enc_output)`:
# 
# * <b>`__init__(self, ...)`: [5 points]</b> Instantiate the parameters of your model, and store them in `self` variables.
# 
# * <b>`compute_attention(self, dec_hs, enc_output)` [5 points]</b>: Compute the <b>context vector</b>, which is a weighted sum of the encoder output states. Suppose the decoder hidden state at time $t$ is $\mathbf{h}_t$, and the encoder hidden state at time $s$ is $\mathbf{\bar h}_s$. The pseudocode is as follows:
# 
#   1. <b>Attention scores:</b> Compute real-valued scores for the decoder hidden state $\mathbf{h}_t$ and each encoder hidden state $\mathbf{\bar h}_s$: $$\mathrm{score}(\mathbf{h}_t, \mathbf{\bar h}_s)=
#       \mathbf{v}_a^T \tanh(\mathbf{W}_1 \mathbf{h}_t +\mathbf{W}_2 \mathbf{\bar h}_s)
# $$
#    Here you should implement the scoring function. A higher score indicates a stronger "affinity" between the decoder state and a specific encoder state. 
#    
#    <font color='green'><b>Hint:</b> the matrices $\mathbf{W}_1$, $\mathbf{W}_2$ and the vector $\mathbf{v_a}$ can all be implemented with `nn.Linear(...)` in Pytorch.</font>
# 
#    Note that in theory, $\mathbf{v_a}$ could have a different dimension than $\mathbf{h}_t$ and $\mathbf{\bar h}_s$, but you should use the same hidden size for this vector.
# 
#  2. <b>Attention weights:</b> Normalize the attention scores to obtain a valid probability distribution: $$\alpha_{ts} = \frac{\exp \big (\mathrm{score}(\mathbf{h}_t, \mathbf{\bar h}_s) \big)}{\sum_{s'=1}^S \exp \big (\mathrm{score}(\mathbf{h}_t, \mathbf{\bar h}_{s'}) \big)}$$ Notice that this is just the softmax function, and can be implemented with `F.softmax(...)` in Pytorch.
# 
#  3. <b>Context vector:</b> Compute a context vector $\mathbf{c}_t$ that is a weighted average of the encoder hidden states, where the weights are given by the attention weights you just computed: $$\mathbf{c}_t=\sum_{s=1}^S \alpha_{ts} \mathbf{\bar h}_s$$
# 
#  You should return this context vector, along with the attention weights.
# 
# 
# 
# * <b>`forward(self, x, dec_hs, enc_output)`: [5 points]</b> Run a <b>single</b> decoding step, resulting in a distribution over the vocabulary for the next token in the sequence. Pseudocode can be found in the docstrings below.
# 
# <font color='green'><b>Hint:</b> You should be able to implement all of this <b>without any for loops</b> using the Pytorch library. Also, remember that these operations should operate in parallel for each item in your batch.</font>

# In[16]:


class RnnDecoder(nn.Module):
    def __init__(self, trg_vocab, embedding_dim, hidden_units):
        super(RnnDecoder, self).__init__()
        """
        Args:
            trg_vocab: Vocab_Lang, the target vocabulary
            embedding_dim: The dimension of the embedding
            hidden_units: The number of features in the GRU hidden state
        """

        self.trg_vocab = trg_vocab # Do not change
        vocab_size = len(trg_vocab)

        ### TODO ###

        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hidden_units = hidden_units
        self.rnn_decoder_softmax = torch.nn.Softmax(dim=1)

        # Initialize embedding layer
        # self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # self.embedding = nn.Embedding(vocab_size, embedding_dim).to(self.device)
        self.embedding = nn.Embedding(vocab_size, embedding_dim).to(device)
        self.embed_size = embedding_dim
        self.num_layers = 1


        # Initialize layers to compute attention score
        # self.w1_dense_layer = nn.Linear(hidden_units, hidden_units)
        # self.w2_dense_layer = nn.Linear(hidden_units, hidden_units)
        # self.v_a_dense_layer = nn.Linear(hidden_units, 1)

        # self.w1_dense_layer = nn.Linear(hidden_units, hidden_units).to(self.device)
        # self.w2_dense_layer = nn.Linear(hidden_units, hidden_units).to(self.device)
        # self.v_a_dense_layer = nn.Linear(hidden_units, 1).to(self.device)

        self.w1_dense_layer = nn.Linear(hidden_units, hidden_units).to(device)
        self.w2_dense_layer = nn.Linear(hidden_units, hidden_units).to(device)
        self.v_a_dense_layer = nn.Linear(hidden_units, 1).to(device)

        # Initialize a single directional GRU with 1 layer and batch_first=True
        # self.GRU = nn.GRU(input_size=embedding_dim+hidden_units, hidden_size=hidden_units, num_layers=self.num_layers, batch_first=True).to(self.device)

        self.GRU = nn.GRU(input_size=embedding_dim+hidden_units, hidden_size=hidden_units, num_layers=self.num_layers, batch_first=True).to(device)

        # NOTE: Input to your RNN will be the concatenation of your embedding vector and the context vector


        # Initialize fully connected layer
        # self.dense_layer = nn.Linear(hidden_units, vocab_size).to(self.device)
        self.dense_layer = nn.Linear(hidden_units, vocab_size).to(device)
        # self.dense_layer = nn.Linear(hidden_units, vocab_size)
    

    def compute_attention(self, dec_hs, enc_output):
        '''
        This function computes the context vector and attention weights.

        Args:
            dec_hs: Decoder hidden state; [1, batch_size, hidden_units]
            enc_output: Encoder outputs; [max_len_src, batch_size, hidden_units]

        Returns:
            context_vector: Context vector, according to formula; [batch_size, hidden_units]
            attention_weights: The attention weights you have calculated; [batch_size, max_len_src, 1]

        Pseudo-code:
            (1) Compute the attention scores for dec_hs & enc_output
                    - Hint: You may need to permute the dimensions of the tensors in order to pass them through linear layers
                    - Output size: [batch_size, max_len_src, 1]
            (2) Compute attention_weights by taking a softmax over your scores to normalize the distribution (Make sure that after softmax the normalized scores add up to 1)
                    - Output size: [batch_size, max_len_src, 1]
            (3) Compute context_vector from attention_weights & enc_output
                    - Hint: You may find it helpful to use torch.sum & element-wise multiplication (* operator)
            (4) Return context_vector & attention_weights
        '''      
        # context_vector, attention_weights = None, None
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # dec_hs = dec_hs
        # enc_output = enc_output
        # dec_hs = dec_hs.to(self.device)
        # enc_output = enc_output.to(self.device)
        dec_hs = dec_hs.to(device)
        enc_output = enc_output.to(device)

        # why is it not even printing anything?
        # print('Content of dec_hs:', dec_hs)
        # print('Shape of dec_hs  [1, batch_size, hidden_units]:', dec_hs.shape, '\n')
        # print('Type of dec_hs:', dec_hs.dtype, '\n')

        # print('Content of enc_output:', enc_output)
        # print('Shape of enc_output [max_len_src, batch_size, hidden_units]:', enc_output.shape, '\n')
        # print('Type of enc_output:', enc_output.dtype, '\n')

        # Shape of dec_hs: torch.Size([1, 1, 50]) == [1, batch_size, hidden_units]
        #
        # Type of dec_hs: torch.float32
        #
        # Shape of enc_output: torch.Size([16, 1, 50])  == [max_len_src, batch_size, hidden_units]
        # batch size = 1, max len src = 16, hidden units = 50
        # Type of enc_output: torch.float32

        # Shape of decoder_hidden_state_ht: torch.Size([1, 1, 1, 50])
        #
        # Shape of encoder_hidden_state_hs: torch.Size([16, 1, 50])

        # Shape of tanh_of_weighted_sum_hidden_states: torch.Size([16, 1, 50])
        #
        # Shape of score Output size: [batch_size, max_len_src, 1]: torch.Size([16, 1, 1])
        #
        # Shape of permuted score Output size: [batch_size, max_len_src, 1]: torch.Size([1, 16, 1])
        # #







        ### TODO ###
        # score = self.v(torch.tanh(self.w1(dec_hs.unsqueeze(1)) + self.w2(enc_output))).squeeze(-1)

        # score = torch.permute(score, (2,1,0))

        # dec_hs: Decoder hidden state;  [1, batch_size, hidden_units]
        # CONVERT to dec_hs: Decoder hidden state; [1, 1, batch_size, hidden_units] ??
        # decoder_hidden_state_ht = self.w1_dense_layer(dec_hs.unsqueeze(1)).to(self.device)
        # decoder_hidden_state_ht = self.w1_dense_layer(dec_hs)
        decoder_hidden_state_ht = self.w1_dense_layer(dec_hs).to(device)
        # print('Content of decoder_hidden_state_ht:', decoder_hidden_state_ht)
        # print('Shape of decoder_hidden_state_ht  [1, batch_size, hidden_units]:', decoder_hidden_state_ht.shape, '\n')
        # print('Type of decoder_hidden_state_ht:', decoder_hidden_state_ht.dtype, '\n')


        #  enc_output: Encoder outputs; [max_len_src, batch_size, hidden_units]
        # CONVERT to enc_output: Encoder outputs; [max_len_src, batch_size, hidden_units]
        # encoder_hidden_state_hs = self.w2_dense_layer(enc_output)
        # encoder_hidden_state_hs = self.w2_dense_layer(enc_output).to(self.device)
        encoder_hidden_state_hs = self.w2_dense_layer(enc_output).to(device)
        # print('Content of encoder_hidden_state_hs:', encoder_hidden_state_hs)
        # print('Shape of encoder_hidden_state_hs [max_len_src, batch_size, hidden_units]:', encoder_hidden_state_hs.shape, '\n')
        # print('Type of encoder_hidden_state_hs:', encoder_hidden_state_hs.dtype, '\n')

        # tanh_of_weighted_sum_hidden_states = torch.tanh(decoder_hidden_state_ht + encoder_hidden_state_hs)
        # tanh_of_weighted_sum_hidden_states = torch.tanh(decoder_hidden_state_ht + encoder_hidden_state_hs).to(self.device)
        tanh_of_weighted_sum_hidden_states = torch.tanh(decoder_hidden_state_ht + encoder_hidden_state_hs).to(device)
        # print('Content of tanh_of_weighted_sum_hidden_states:', tanh_of_weighted_sum_hidden_states)
        # print('Shape of tanh_of_weighted_sum_hidden_states:', tanh_of_weighted_sum_hidden_states.shape, '\n')
        # print('Type of tanh_of_weighted_sum_hidden_states:', tanh_of_weighted_sum_hidden_states.dtype, '\n')

        score = self.v_a_dense_layer(tanh_of_weighted_sum_hidden_states)  # transpose????
        # print('Content of score:', score)
        # print('Shape of score Output size: [batch_size, max_len_src, 1]:', score.shape, '\n')
        # print('Type of score:', score.dtype, '\n')




        score = torch.permute(score, (1,0,2))
        # print('Content of score:', score)
        # print('Shape of score Output size: [batch_size, max_len_src, 1]:', score.shape, '\n')
        # Shape of permuted score Output size: [batch_size, max_len_src, 1]: torch.Size([1, 16, 1])
        # print('Type of score:', score.dtype, '\n')

        # (2) Compute attention_weights by taking a softmax over your scores to normalize the distribution (Make sure that after softmax the normalized scores add up to 1)
        #             - Output size: [batch_size, max_len_src, 1]
        #softmax
        attention_weights_alpha_ts = self.rnn_decoder_softmax(score)
        # print('Content of attention_weights_alpha_ts:', attention_weights_alpha_ts)
        # print('Shape of attention_weights_alpha_ts Output size: [batch_size, max_len_src, 1]:', attention_weights_alpha_ts.shape, '\n')
        # print('Type of score:', score.dtype, '\n')
        # Shape of attention_weights_alpha_ts Output size: [batch_size, max_len_src, 1]: torch.Size([1, 16, 1])

        # check sum
        sum_check = torch.sum(attention_weights_alpha_ts)
        # print('Content of sum_check:', sum_check)

        # (3) Compute context_vector from attention_weights & enc_output
        #             - Hint: You may find it helpful to use torch.sum & element-wise multiplication (* operator)

        # Shape of attention_weights_alpha_ts Output size: [batch_size, max_len_src, 1]: torch.Size([1, 16, 1])
        # Shape of enc_output: torch.Size([16, 1, 50])  == [max_len_src, batch_size, hidden_units]
        # context_vector = torch.bmm(torch.permute(attention_weights_alpha_ts, (0,1,2)), torch.permute(enc_output, (1,0,2)))

        # attention_weights_alpha_ts_squeezed = attention_weights_alpha_ts.squeeze(-1)  #1, 16 = batch, max len
        attention_weights_alpha_ts_squeezed = attention_weights_alpha_ts  #1, 16, 1 = batch, max len, 1
        # print('Shape of attention_weights_alpha_ts_squeezed #1, 16 = batch, max len:', attention_weights_alpha_ts_squeezed.shape, '\n')

        enc_output_permuted = torch.permute(enc_output, (1,0,2)) #1, 16, 50 = batch, max_len, hidden units
        # print('Shape of enc_output_permuted #1, 16, 50 = batch, max_len, hidden units:', enc_output_permuted.shape, '\n')

        # enc_output_permuted_sum = torch.sum(enc_output_permuted, axis=2)
        # print('Shape of enc_output_permuted_sum Output size: [batch_size, max_len_src]:', enc_output_permuted_sum.shape, '\n')

        # context_vector = attention_weights_alpha_ts_squeezed * enc_output_permuted_sum
        # print('Shape of context_vector [batch_size, hidden_units] == [1, 50]:', context_vector.shape, '\n')


        context_vector_element_wise_multiplication = attention_weights_alpha_ts_squeezed * enc_output_permuted
        # print('Shape of context_vector [batch_size, hidden_units] == [1, 50]:', context_vector_element_wise_multiplication.shape, '\n')

        context_vector = torch.sum(context_vector_element_wise_multiplication, axis=1)
        # print('Shape of context_vector [batch_size, hidden_units] == [1, 50]:', context_vector.shape, '\n')


        # context_vector = [batch_size, hidden_units] == [1, 50]

        attention_weights = attention_weights_alpha_ts.type(torch.float32)

        return context_vector, attention_weights

    def forward(self, x, dec_hs, enc_output):
        '''
        This function runs the decoder for a **single** time step.

        Args:
            x: Input token; [batch_size, 1]
            dec_hs: Decoder hidden state; [1, batch_size, hidden_units]
            enc_output: Encoder outputs; [max_len_src, batch_size, hidden_units]

        Returns:
            fc_out: (Unnormalized) output distribution [batch_size, vocab_size]
            dec_hs: Decoder hidden state; [1, batch_size, hidden_units]
            attention_weights: The attention weights you have learned; [batch_size, max_len_src, 1]

        Pseudo-code:
            (1) Compute the context vector & attention weights by calling self.compute_attention(...) on the appropriate input
            (2) Obtain embedding vectors for your input x
                    - Output size: [batch_size, 1, embedding_dim]             
            (3) Concatenate the context vector & the embedding vectors along the appropriate dimension
            (4) Feed this result through your RNN (along with the current hidden state) to get output and new hidden state
                    - Output sizes: [batch_size, 1, hidden_units] & [1, batch_size, hidden_units] 
            (5) Feed the output of your RNN through linear layer to get (unnormalized) output distribution (don't call softmax!)
            (6) Return this output, the new decoder hidden state, & the attention weights
        '''
        fc_out, attention_weights = None, None

        # print(f"FORWARD of DECODER: x.shape: {x.shape} \n, dec_hs.shape: {dec_hs.shape} \n, enc_output.shape: {enc_output.shape}")
        ### TODO ###
        context_vector, attention_weights = self.compute_attention(dec_hs, enc_output)
        # print('FORWARD of DECODER: Shape of context_vector [batch_size, hidden_units] == [1, 50]:', context_vector.shape, '\n')

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        x = x.to(device)
        final_embedding =  self.embedding(x)

        final_embedding_gpu = final_embedding.to(device)
        # final_embedding_gpu = final_embedding
        embedding_vector = final_embedding_gpu
        # print(f"embedding dimn:{embedding_dim}")
        # print('Shape of embedding_vector  [batch_size, 1, embedding_dim] == [1, 1, 50]:', embedding_vector.shape, '\n')

        context_vector_unsqueezed = (context_vector.unsqueeze(1)).to(device)
        # print('Shape of context_vector_unsqueezed [batch_size, 1, hidden_units] == [1, 1, 50]::', context_vector_unsqueezed.shape, '\n')

        # concatenated_tensor = (torch.cat([context_vector_unsqueezed, embedding_vector], dim=2))
        concatenated_tensor = (torch.cat([context_vector_unsqueezed, embedding_vector], dim=2)).to(device)
        # print(f"  concatenated_tensor shape: : {concatenated_tensor.shape}")

        rnn_output, hn  = self.GRU(concatenated_tensor)

        # [batch_size, 1, hidden_units] & [1, batch_size, hidden_units]
        # print(f"  rnn_output: {rnn_output.shape}")
        # print(f" hn: {hn.shape}")

        # linear_output = self.dense_layer(rnn_output)
        linear_output = self.dense_layer(rnn_output).to(device)
        # print(f"  linear_output: {linear_output.shape}")

        # linear_output_squeezed = (linear_output.squeeze(1))
        linear_output_squeezed = (linear_output.squeeze(1)).to(device)
        # print(f"  linear_output_squeezed: {linear_output_squeezed.shape}")

        fc_out = linear_output_squeezed
        dec_hs = hn
        attention_weights = attention_weights


        return fc_out, dec_hs, attention_weights


# ## Sanity Check: RNN Decoder Model
# 
# The code below runs a sanity check for your `RnnDecoder` class. The tests are similar to the hidden ones in Gradescope. However, note that passing the sanity check does <b>not</b> guarantee that you will pass the autograder; it is intended to help you debug.

# In[17]:


### DO NOT EDIT ###

def sanityCheckDecoderModelForward(inputs, NN, expected_outputs):
    print('--- TEST: Output shape of forward(...) ---\n')
    expected_fc_outs = expected_outputs[0]
    expected_dec_hs = expected_outputs[1]
    expected_attention_weights = expected_outputs[2]
    msg = ''
    for i, inp in enumerate(inputs):
        input_rep = '{'
        for k,v in inp.items():
            if torch.is_tensor(v):
                input_rep += str(k) + ': ' + 'Tensor with shape ' + str(v.size()) + ', '
            else:
                input_rep += str(k) + ': ' + str(v) + ', '
        input_rep += '}'
        dec = RnnDecoder(trg_vocab=inp['trg_vocab'],embedding_dim=inp['embedding_dim'],hidden_units=inp['hidden_units'])
        dec_hs = torch.rand(1, inp["batch_size"], inp['hidden_units'])
        x = torch.randint(low=0,high=len(inp["trg_vocab"]),size=(inp["batch_size"], 1))
        with torch.no_grad():
            dec_out = dec(x=x, dec_hs=dec_hs,enc_output=inp['encoder_outputs'])
            if not isinstance(dec_out, tuple):
                msg = '\tFAILED\tYour RnnDecoder.forward() output must be a tuple; received ' + str(type(dec_out))
                print(msg)
                continue
            elif len(dec_out)!=3:
                msg = '\tFAILED\tYour RnnDecoder.forward() output must be a tuple of size 3; received tuple of size ' + str(len(dec_out))
                print(msg)
                continue
            stu_fc_out, stu_dec_hs, stu_attention_weights = dec_out
        del dec
        has_passed = True
        msg = ""
        if not torch.is_tensor(stu_fc_out):
            has_passed = False
            msg += '\tFAILED\tOutput must be a torch.Tensor; received ' + str(type(stu_fc_out)) + " "
        if not torch.is_tensor(stu_dec_hs):
            has_passed = False
            msg += '\tFAILED\tDecoder Hidden State must be a torch.Tensor; received ' + str(type(stu_dec_hs)) + " "
        if not torch.is_tensor(stu_attention_weights):
            has_passed = False
            msg += '\tFAILED\tAttention Weights must be a torch.Tensor; received ' + str(type(stu_attention_weights)) + " "

        status = 'PASSED' if has_passed else 'FAILED'
        if not has_passed:
            message = '\t' + status + "\t Init Input: " + input_rep + '\tForward Input Shape: ' + str(inp['encoder_outputs'].shape) + '\tExpected Output Shape: ' + str(expected_fc_outs[i]) + '\t' + msg
            print(message)
            continue

        has_passed = stu_fc_out.size() == expected_fc_outs[i]
        msg = 'Your Output Shape: ' + str(stu_fc_out.size())
        status = 'PASSED' if has_passed else 'FAILED'
        message = '\t' + status + "\t Init Input: " + input_rep + '\tForward Input Shape: ' + str(inp['encoder_outputs'].shape) + '\tExpected Output Shape: ' + str(expected_fc_outs[i]) + '\t' + msg
        print(message)

        has_passed = stu_dec_hs.size() == expected_dec_hs[i]
        msg = 'Your Hidden State Shape: ' + str(stu_dec_hs.size())
        status = 'PASSED' if has_passed else 'FAILED'
        message = '\t' + status + "\t Init Input: " + input_rep + '\tForward Input Shape: ' + str(inp['encoder_outputs'].shape) + '\tExpected Hidden State Shape: ' + str(expected_dec_hs[i]) + '\t' + msg
        print(message)

        has_passed = stu_attention_weights.size() == expected_attention_weights[i]
        msg = 'Your Attention Weights Shape: ' + str(stu_attention_weights.size())
        status = 'PASSED' if has_passed else 'FAILED'
        message = '\t' + status + "\t Init Input: " + input_rep + '\tForward Input Shape: ' + str(inp['encoder_outputs'].shape) + '\tExpected Attention Weights Shape: ' + str(expected_attention_weights[i]) + '\t' + msg
        print(message)

        stu_sum = stu_attention_weights.sum(dim=1).squeeze()
        if torch.allclose(stu_sum, torch.ones_like(stu_sum), atol=1e-5):
            print('\tPASSED\t The sum of your attention_weights along dim 1 is 1.')
        else:
            print('\tFAILED\t The sum of your attention_weights along dim 1 is not 1.')
        print()


# In[18]:


### DO NOT EDIT ###

if __name__ == '__main__':
    # Set random seed
    torch.manual_seed(42)
    # Create test inputs
    embedding_dim = [2, 5, 8]
    hidden_units = [50, 100, 200]
    params = []
    inputs = []
    for i in range(len(embedding_dim)):
        for hu in hidden_units:
            inp = {}
            inp['trg_vocab'] = trg_vocab
            inp['embedding_dim'] = embedding_dim[i]
            inp['hidden_units'] = hu
            inputs.append(inp)
    # Test init
    expected_outputs = [371028, 762228, 1664628, 391305, 782955, 1686255, 411582, 803682, 1707882]
    sanityCheckModel(inputs, RnnDecoder, expected_outputs, "init", None)
    print()

    # Test forward
    inputs = []
    batch_sizes = [1, 2, 4]
    embedding_dims = iter([50,80,100,120,150,200,300,400,500])
    encoder_outputs = iter([torch.rand([1, 16, 50]), torch.rand([2, 16, 50]), torch.rand([4, 16, 50]), torch.rand([1, 16, 100]), torch.rand([2, 16, 100]), torch.rand([4, 16, 100]), torch.rand([1, 16, 200]), torch.rand([2, 16, 200]),torch.rand([4, 16, 200])])
    expected_fc_outs = [torch.Size([1, 6609]),torch.Size([2, 6609]),torch.Size([4, 6609]),torch.Size([1, 6609]),torch.Size([2, 6609]),torch.Size([4, 6609]),torch.Size([1, 6609]),torch.Size([2, 6609]),torch.Size([4, 6609])]
    expected_dec_hs = [torch.Size([1, 1, 50]), torch.Size([1, 2, 50]), torch.Size([1, 4, 50]), torch.Size([1, 1, 100]), torch.Size([1, 2, 100]), torch.Size([1, 4, 100]), torch.Size([1, 1, 200]), torch.Size([1, 2, 200]), torch.Size([1, 4, 200])]
    expected_attention_weights = [torch.Size([1, 16, 1]), torch.Size([2, 16, 1]), torch.Size([4, 16, 1]), torch.Size([1, 16, 1]), torch.Size([2, 16, 1]), torch.Size([4, 16, 1]), torch.Size([1, 16, 1]), torch.Size([2, 16, 1]), torch.Size([4, 16, 1])]
    expected_outputs = (expected_fc_outs, expected_dec_hs, expected_attention_weights)
    
    for hu in hidden_units:
        for b in batch_sizes:
            inp = {}
            edim = next(embedding_dims)
            inp['embedding_dim'] = edim
            inp['trg_vocab'] = trg_vocab
            inp["batch_size"] = b
            inp['hidden_units'] = hu
            inp['encoder_outputs'] = next(encoder_outputs).transpose(0,1)
            inputs.append(inp)
    
    sanityCheckDecoderModelForward(inputs, RnnDecoder, expected_outputs)


# ## Train RNN Model
# 
# We will train the encoder and decoder using cross-entropy loss.

# In[19]:


### DO NOT EDIT ###

def loss_function(real, pred):
    mask = real.ge(1).float() # Only consider non-zero inputs in the loss
    
    loss_ = F.cross_entropy(pred, real) * mask 
    return torch.mean(loss_)

def train_rnn_model(encoder, decoder, dataset, optimizer, trg_vocab, device, n_epochs):
    batch_size = dataset.batch_size
    for epoch in range(n_epochs):
        start = time.time()
        n_batch = 0
        total_loss = 0
        
        encoder.train()
        decoder.train()
        
        for src, trg in tqdm(dataset):
            n_batch += 1
            loss = 0
            
            enc_output, enc_hidden = encoder(src.transpose(0,1).to(device))
            dec_hidden = enc_hidden
            
            # use teacher forcing - feeding the target as the next input (via dec_input)
            dec_input = torch.tensor([[trg_vocab.word2idx['<start>']]] * batch_size)
        
            # run code below for every timestep in the ys batch
            for t in range(1, trg.size(1)):
                predictions, dec_hidden, _ = decoder(dec_input.to(device), dec_hidden.to(device), enc_output.to(device))
                assert len(predictions.shape) == 2 and predictions.shape[0] == dec_input.shape[0] and predictions.shape[1] == len(trg_vocab.word2idx), "First output of decoder must have shape [batch_size, vocab_size], you returned shape " + str(predictions.shape)
                loss += loss_function(trg[:, t].to(device), predictions.to(device))
                dec_input = trg[:, t].unsqueeze(1)
        
            batch_loss = (loss / int(trg.size(1)))
            total_loss += batch_loss
            
            optimizer.zero_grad()
            
            batch_loss.backward()

            ### update model parameters
            optimizer.step()
        
        ### TODO: Save checkpoint for model (optional)
        print('Epoch:{:2d}/{}\t Loss: {:.4f} \t({:.2f}s)'.format(epoch + 1, n_epochs, total_loss / n_batch, time.time() - start))

    print('Model trained!')


# In[20]:


### DO NOT EDIT ###

if __name__ == '__main__':
    # HYPERPARAMETERS - feel free to change
    LEARNING_RATE = 0.001
    HIDDEN_UNITS=256
    N_EPOCHS=1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
    rnn_encoder = RnnEncoder(src_vocab, EMBEDDING_DIM, HIDDEN_UNITS).to(device)
    rnn_decoder = RnnDecoder(trg_vocab, EMBEDDING_DIM, HIDDEN_UNITS).to(device)

    rnn_model_params = list(rnn_encoder.parameters()) + list(rnn_decoder.parameters())
    optimizer = torch.optim.Adam(rnn_model_params, lr=LEARNING_RATE)

    print('Encoder and Decoder models initialized!')


# In[21]:


### DO NOT EDIT ###

if __name__ == '__main__':
    train_rnn_model(rnn_encoder, rnn_decoder, train_dataset, optimizer, trg_vocab, device, N_EPOCHS)


# ## <font color='red'>TODO:</font> Inference (Decoding) Function [5 points]
# 
# Now that we have trained the model, we can use it on test data.
# 
# Here, you will write a function that takes your trained model and a source sentence (Spanish), and returns its translation (English sentence). Instead of using teacher forcing, the input to the decoder at time step $t_i$ will be the prediction of the decoder at time $t_{i-1}$.

# In[22]:


def decode_rnn_model(encoder, decoder, src, max_decode_len, device):
    """
    Args:
        encoder: Your RnnEncoder object
        decoder: Your RnnDecoder object
        src: [max_src_length, batch_size] the source sentences you wish to translate
        max_decode_len: The maximum desired length (int) of your target translated sentences
        device: the device your torch tensors are on (you may need to call x.to(device) for some of your tensors)

    Returns:
        curr_output: [batch_size, max_decode_len] containing your predicted translated sentences
        curr_predictions: [batch_size, max_decode_len, trg_vocab_size] containing the (unnormalized) probabilities of each
            token in your vocabulary at each time step

    Pseudo-code:
    - Obtain encoder output and hidden state by encoding src sentences
    - For 1 ≤ t ≤ max_decode_len:
        - Obtain your (unnormalized) prediction probabilities and hidden state by feeding dec_input (the best words 
          from the previous time step), previous hidden state, and encoder output to decoder
        - Save your (unnormalized) prediction probabilities in curr_predictions at index t
        - Obtain your new dec_input by selecting the most likely (highest probability) token
        - Save dec_input in curr_output at index t
    """
    # Initialize variables
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trg_vocab = (decoder.trg_vocab)
    batch_size = src.size(1)

    curr_output = (torch.zeros((batch_size, max_decode_len))).to(device)
    curr_predictions = (torch.zeros((batch_size, max_decode_len, len(trg_vocab.idx2word)))).to(device)

    # We start the decoding with the start token for each example
    dec_input = (torch.tensor([[trg_vocab.word2idx['<start>']]] * batch_size)).to(device)
    # print(f"OUTSIDE FOR LOOP dec_input: {dec_input}, dec_input.type = {dec_input.type}")
    # print(f"OUTSIDE FOR LOOP dec_input.shape: {dec_input.shape}")

    # dec_input_unsqueezed = (dec_input.unsqueeze(1)).to(device)
    curr_output[:, 0] = dec_input.squeeze(1)
    
    ### TODO: Implement decoding algorithm ###
    # print(f"INFERENCE of DECODER src:{src}  [max_src_length, batch_size] src.shape: {src.shape} \n")
    encoder_output, encoder_hidden_state = encoder(src)
    encoder_output = encoder_output.to(device)
    encoder_hidden_state = encoder_hidden_state.to(device)

    # print(f"max_decode_len: {max_decode_len}")
    dec_hs = encoder_hidden_state.to(device)


    for t in range(1,max_decode_len):
        # print(f"INSIDE FOR LOOP T is: {t}")
        if t==0:
            x = dec_input.to(device)
            # print(f"x when t={t}: {x}, x.shape={x.shape}")
        elif t>0:
            x_int = (dec_input[:, -1]).to(device)
            # print(f"x when t={t}: {x}, x.shape={x.shape}")
            x = x_int.view(batch_size,1).to(device)
            # print(f"x after view when t={t}: {x}, x.shape={x.shape}")

        # dec_input_unsqueezed = (dec_input.unsqueeze(1)).to(device)
        #
        # x = (dec_input_unsqueezed).to(device)
        # x = (dec_input).to(device)

        # x = (target).to(device)



        # if t==0:
        #     dec_hs = encoder_hidden_state.to(device)
        #
        # elif t>0:
        #     dec_hs=dec_hs
        # print('Shape of dec_hs dec_hs: Decoder hidden state; [1, batch_size, hidden_units]:', dec_hs.shape, '\n')

        enc_output = encoder_output.to(device)
        # print('Shape of enc_output Encoder outputs; [max_len_src, batch_size, hidden_units]:', enc_output.shape, '\n')

        # print(f"INFERENCE of DECODER: [batch_size, 1] x.shape: {x.shape} \n, dec_hs.shape: {dec_hs.shape} \n, enc_output.shape: {enc_output.shape}")
        # print('Shape of dec_input :', dec_input.shape, '\n')
        # print('Shape of dec_input_unsqueezed :', dec_input_unsqueezed.shape, '\n')
        # print('Shape of x [batch_size, 1] :', x.shape, '\n')








        # x.shape: torch.Size([4, 1]),
        # , dec_hs.shape: torch.Size([1, 4, 200]),
        # , enc_output.shape: torch.Size([16, 4, 200])
        # Shape of context_vector [batch_size, hidden_units] == [1, 50]: torch.Size([4, 200])

        # Shape of dec_input : torch.Size([5, 1])
        #
        # Shape of x : torch.Size([5, 1])
        #
        # Shape of dec_hs : torch.Size([1, 5, 256])
        #
        # Shape of enc_output : torch.Size([16, 5, 256])

        """
         Args:
            x: Input token; [batch_size, 1]
            dec_hs: Decoder hidden state; [1, batch_size, hidden_units]
            enc_output: Encoder outputs; [max_len_src, batch_size, hidden_units]
        """
        fc_out, dec_hs, attention_weights = decoder(x=x, dec_hs=dec_hs,enc_output=enc_output)
        fc_out = fc_out.to(device)
        dec_hs = dec_hs.to(device)
        attention_weights = attention_weights.to(device)
        # print('Shape of fc_out  (Unnormalized) output distribution [batch_size, vocab_size]:', fc_out.shape)
        # print('Shape of dec_hs Decoder hidden state; [1, batch_size, hidden_units]:', dec_hs.shape)
        # print('Shape of attention_weights [batch_size, max_len_src, 1]:', attention_weights.shape)


        curr_predictions[:, t, :] = fc_out.to(device)
        # print('Shape of curr_predictions : [batch_size, max_decode_len, trg_vocab_size]:', curr_predictions.shape, '\n')

        new_dec_input = (torch.argmax(fc_out, dim=1)).to(device)
        # print('Shape of new_dec_input:', new_dec_input.shape, '\n')

        new_dec_input_unsqueezed = (torch.argmax(fc_out, dim=1).unsqueeze(1)).to(device)
        # print('Shape of new_dec_input_unsqueezed:', new_dec_input_unsqueezed.shape, '\n')

        # print('Shape of dec_input before concatenation:', dec_input.shape, '\n')
        dec_input = (torch.cat([dec_input, new_dec_input_unsqueezed], dim=1)).to(device)
        # print('Shape of dec_input after concatenation:', dec_input.shape, '\n')
        # print(f"dec_input: {dec_input}")

        # dec_input.append(new_dec_input)
        # dec_input = ((torch.argmax(fc_out, dim=1)).unsqueeze(1)).to(device)
        # print('new_dec_input:', new_dec_input)


        curr_output[:, t] = new_dec_input
        # print('curr_output:', curr_output)
        # print('Shape of curr_output: [batch_size, max_decode_len]:', curr_output.shape, '\n')



    return curr_output, curr_predictions


# You can run the cell below to qualitatively compare some of the sentences your model generates with the some of the correct translations.

# In[23]:


### DO NOT EDIT ###

if __name__ == '__main__':
    rnn_encoder.eval()
    rnn_decoder.eval()
    idxes = random.choices(range(len(test_dataset.dataset)), k=5)
    src, trg =  train_dataset.dataset[idxes]
    curr_output, _ = decode_rnn_model(rnn_encoder, rnn_decoder, src.transpose(0,1).to(device), trg.size(1), device)
    for i in range(len(src)):
        print("Source sentence:", ' '.join([x for x in [src_vocab.idx2word[j.item()] for j in src[i]] if x != '<pad>']))
        print("Target sentence:", ' '.join([x for x in [trg_vocab.idx2word[j.item()] for j in trg[i]] if x != '<pad>']))
        print("Predicted sentence:", ' '.join([x for x in [trg_vocab.idx2word[j.item()] for j in curr_output[i]] if x != '<pad>']))
        print("----------------")


# ## Evaluate RNN Model [20 points]
# 
# We provide you with a function to run the test set through the model and calculate BLEU scores. We expect your BLEU scores to satisfy the following conditions:  
# 
# *   BLEU-1 > 0.290
# *   BLEU-2 > 0.082
# *   BLEU-3 > 0.060
# *   BLEU-4 > 0.056
# 
# Read more about Bleu Score at :
# 
# 1.   https://en.wikipedia.org/wiki/BLEU
# 2.   https://www.aclweb.org/anthology/P02-1040.pdf

# In[24]:


### DO NOT EDIT ###

def get_reference_candidate(target, pred, trg_vocab):
    def _to_token(sentence):
        lis = []
        for s in sentence[1:]:
            x = trg_vocab.idx2word[s]
            if x == "<end>": break
            lis.append(x)
        return lis
    reference = _to_token(list(target.numpy()))
    candidate = _to_token(list(pred.numpy()))
    return reference, candidate

def compute_bleu_scores(target_tensor_val, target_output, final_output, trg_vocab):
    bleu_1 = 0.0
    bleu_2 = 0.0
    bleu_3 = 0.0
    bleu_4 = 0.0

    smoother = SmoothingFunction()
    save_reference = []
    save_candidate = []
    for i in range(len(target_tensor_val)):
        reference, candidate = get_reference_candidate(target_output[i], final_output[i], trg_vocab)
    
        bleu_1 += sentence_bleu(reference, candidate, weights=(1,), smoothing_function=smoother.method1)
        bleu_2 += sentence_bleu(reference, candidate, weights=(1/2, 1/2), smoothing_function=smoother.method1)
        bleu_3 += sentence_bleu(reference, candidate, weights=(1/3, 1/3, 1/3), smoothing_function=smoother.method1)
        bleu_4 += sentence_bleu(reference, candidate, weights=(1/4, 1/4, 1/4, 1/4), smoothing_function=smoother.method1)

        save_reference.append(reference)
        save_candidate.append(candidate)
    
    bleu_1 = bleu_1/len(target_tensor_val)
    bleu_2 = bleu_2/len(target_tensor_val)
    bleu_3 = bleu_3/len(target_tensor_val)
    bleu_4 = bleu_4/len(target_tensor_val)

    scores = {"bleu_1": bleu_1, "bleu_2": bleu_2, "bleu_3": bleu_3, "bleu_4": bleu_4}
    print('BLEU 1-gram: %f' % (bleu_1))
    print('BLEU 2-gram: %f' % (bleu_2))
    print('BLEU 3-gram: %f' % (bleu_3))
    print('BLEU 4-gram: %f' % (bleu_4))

    return save_candidate, scores

def evaluate_rnn_model(encoder, decoder, test_dataset, target_tensor_val, device):
    trg_vocab = decoder.trg_vocab
    batch_size = test_dataset.batch_size
    n_batch = 0
    total_loss = 0

    encoder.eval()
    decoder.eval()
    
    final_output, target_output = None, None

    with torch.no_grad():
        for batch, (src, trg) in enumerate(test_dataset):
            n_batch += 1
            loss = 0
            curr_output, curr_predictions = decode_rnn_model(encoder, decoder, src.transpose(0,1).to(device), trg.size(1), device)
            for t in range(1, trg.size(1)):
                loss += loss_function(trg[:, t].to(device), curr_predictions[:,t,:].to(device))

            if final_output is None:
                final_output = torch.zeros((len(target_tensor_val), trg.size(1)))
                target_output = torch.zeros((len(target_tensor_val), trg.size(1)))
            final_output[batch*batch_size:(batch+1)*batch_size] = curr_output
            target_output[batch*batch_size:(batch+1)*batch_size] = trg
            batch_loss = (loss / int(trg.size(1)))
            total_loss += batch_loss

        print('Loss {:.4f}'.format(total_loss / n_batch))
    
    # Compute BLEU scores
    return compute_bleu_scores(target_tensor_val, target_output, final_output, trg_vocab)


# In[25]:


### DO NOT EDIT ###

if __name__ == '__main__':
    rnn_save_candidate, rnn_scores = evaluate_rnn_model(rnn_encoder, rnn_decoder, test_dataset, trg_tensor_val, device)


# # Step 3: Train a Transformer [45 points]
# 
# Here you will write a transformer model for machine translation, and then train and evaluate its results. Here are some helpful links:
# <ul>
# <li> Original transformer paper: https://arxiv.org/pdf/1706.03762.pdf
# <li> Helpful tutorial: http://jalammar.github.io/illustrated-transformer/
# <li> Another tutorial: http://peterbloem.nl/blog/transformers
# </ul>

# In[26]:


### DO NOT EDIT ###

import math


# ## <font color='red'>TODO:</font> Positional Embeddings [5 points]
# 
# Similar to the RNN, we start with the Encoder model. A key component of the encoder is the Positional Embedding. As we know, word embeddings encode words in such a way that words with similar meaning have similar vectors. Because there are no recurrences in a Transformer, we need a way to tell the transformer the relative position of words in a sentence: so will add a positional embedding to the word embeddings. Now, two words with a similar embedding will both be close in meaning and occur near each other in the sentence.
# 
# You will create a positional embedding matrix of size $(max\_len, embed\_dim)$ using the following formulae:
# <br>
# $\begin{align*} pe[pos,2i] &= \sin \Big (\frac{pos}{10000^{2i/embed\_dim}}\Big )\\pe[pos,2i+1] &= \cos \Big (\frac{pos}{10000^{2i/embed\_dim}}\Big ) \end{align*}$
# <font color='green'><b>Hint:</b> You should probably take the logarithm of the denominator to avoid raising $10000$ to an exponent and then exponentiate the result before plugging it into the fraction. This will help you avoid numerical (overflow/underflow) issues.
# 
# <font color='green'><b>Hint:</b> We encourage you to try to implement this function with no for loops, which is the general practice (as it is faster). However, since we are using relatively small datasets, you are welcome to do this with for loops if you prefer.

# In[27]:


def create_positional_embedding(max_len, embed_dim):
    '''
    Args:
        max_len: The maximum length supported for positional embeddings
        embed_dim: The size of your embeddings
    Returns:
        pe: [max_len, 1, embed_dim] computed as in the formulae above
    '''
    pe = None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ### TODO ###
    n = 100
    n_tensor = torch.LongTensor([n])
    d = embed_dim

    # pe = torch.zeros((max_len, d))
    # positional_encoding_matrix = np.zeros((max_len, d))
    positional_encoding_matrix = torch.zeros((max_len, d))

    for token_index in range(max_len):
        # for map_column_indices_i in range(int(d/2)):
        for map_column_indices_i in np.arange(int(d/2)):

            exponent = torch.FloatTensor([float(2*map_column_indices_i/d)])
            # print(f"exponent: {exponent}, exponent.shape: {exponent.shape}")

            ########################################################################
            # NUMPY
            # denominator = np.power(n, 2*i/d)
            # positional_encoding_matrix[token_index, 2*map_column_indices_i] = np.sin(token_index/denominator)
            # positional_encoding_matrix[token_index, 2*map_column_indices_i+1] = np.cos(token_index/denominator)

            #########################################################################
            # PYTORCH
            denominator = torch.pow(n_tensor, exponent)
            # print(f"denominator: {denominator}, denominator.shape: {denominator.shape}")
            positional_encoding_matrix[token_index, 2*map_column_indices_i] = torch.sin(torch.FloatTensor(token_index/denominator))
            positional_encoding_matrix[token_index, (2*map_column_indices_i)+1] = torch.cos(torch.FloatTensor(token_index/denominator))

    # print(f"positional encoding is: {positional_encoding_matrix}. positional_encoding_matrix.shape is: {positional_encoding_matrix.shape}")
    # print(f"positional_encoding_matrix.shape is: {positional_encoding_matrix.shape}")

    pe_tensor = torch.FloatTensor(positional_encoding_matrix).to(device)
    # print(f"positional encoding is: {pe_tensor}. pe.shape is: {pe_tensor.shape}")
    # print(f"pe_tensor.shape is: {pe_tensor.shape}")

    pe_tensor_unsquezed = pe_tensor.unsqueeze(1)
    # print(f"positional encoding unsqueezed is: {pe_tensor_unsquezed}. pe_tensor_unsquezed.shape is: {pe_tensor_unsquezed.shape}")
    # print(f"pe_tensor_unsquezed.shape is: {pe_tensor_unsquezed.shape}")

    return pe_tensor_unsquezed


# In[28]:



# import numpy as np
# import matplotlib.pyplot as plt
#
# def getPositionEncoding(seq_len, d, n=10000):
#     P = np.zeros((seq_len, d))
#     for k in range(seq_len):
#         for i in np.arange(int(d/2)):
#             denominator = np.power(n, 2*i/d)
#             P[k, 2*i] = np.sin(k/denominator)
#             P[k, 2*i+1] = np.cos(k/denominator)
#     return P
#
# P = getPositionEncoding(seq_len=4, d=4, n=100)
# print(P)


# In[29]:


# pe_tensor_unsquezed = create_positional_embedding(max_len=4, embed_dim=4)
# print(pe_tensor_unsquezed)
# print(pe_tensor_unsquezed[0,0,2])


# ## <font color='red'>TODO:</font> Encoder Model [10 points]
# 
# Now you will create the Encoder model for the transformer.
# 
# In this cell, you should implement the `__init(...)` and `forward(...)` functions, each of which is <b>5 points</b>.

# In[30]:


class TransformerEncoder(nn.Module):
    def __init__(self, src_vocab, embedding_dim, num_heads, num_layers, dim_feedforward, max_len_src, device):
        super(TransformerEncoder, self).__init__()
        self.device = device
        """
        Args:
            src_vocab: Vocab_Lang, the source vocabulary,
            embedding_dim: the dimension of the embedding (also the number of expected features for the input of the Transformer)
            num_heads: The number of attention heads
            num_layers: the number of Transformer Encoder layers
            dim_feedforward: the dimension of the feedforward network models in the Transformer
            max_len_src: maximum length of the source sentences
            device: the working device (you may need to map your positional embedding to this device)
        """
        self.src_vocab = src_vocab # Do not change
        src_vocab_size = len(src_vocab)

        # Create positional embedding matrix
        self.position_embedding = create_positional_embedding(max_len_src, embedding_dim).to(device)
        self.register_buffer('positional_embedding', self.position_embedding) # this informs the model that position_embedding is not a learnable parameter

        ### TODO ###

        # Initialize embedding layer
        self.embedding = nn.Embedding(src_vocab_size, embedding_dim).to(device)

        # Dropout layer
        self.dropout_layer = (nn.Dropout(0.5)).to(device)

        # Initialize a nn.TransformerEncoder model (you'll need to use embedding_dim,
        # num_layers, num_heads, & dim_feedforward here)

        # encoder_layer – an instance of the TransformerEncoderLayer() class (required).
        #
        # num_layers – the number of sub-encoder-layers in the encoder (required).
        #
        # norm – the layer normalization component (optional).
        #
        # enable_nested_tensor – if True, input will automatically convert to nested tensor (and convert back on output). This will improve the overall performance of TransformerEncoder when padding rate is high. Default: True (enabled).


        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, dim_feedforward= dim_feedforward, nhead=num_heads ).to(device)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers).to(device)

        # src = torch.rand()
        # src = torch.rand(10, 32, 512)
        # out = transformer_encoder(src)


    def make_src_mask(self, src):
        """
        Args:
            src: [max_len, batch_size]
        Returns:
            Boolean matrix of size [batch_size, max_len] indicating which indices are padding
        """
        assert len(src.shape) == 2, 'src must have exactly 2 dimensions'
        src_mask = src.transpose(0, 1) == 0 # padding idx
        return src_mask.to(self.device) # [batch_size, max_src_len]

    def forward(self, x):
        """
        Args:
            x: [max_len, batch_size]
        Returns:
            output: [max_len, batch_size, embed_dim]
        Pseudo-code (note: x refers to the original input to this function throughout the pseudo-code):
        - Pass x through the word embedding
        - Add positional embedding to the word embedding, then apply dropout
        - Call make_src_mask(x) to compute a mask: this tells us which indexes in x
          are padding, which we want to ignore for the self-attention
        - Call the encoder, with src_key_padding_mask = src_mask
        """
        output = None

        ### TODO ###
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # x_int64 = x.type(torch.int64)

        # print('Content of embedding:', x_int64)
        # print('Shape of embedding:', x_int64.shape, '\n')
        # print('Type of embedding:', x_int64.dtype, '\n')

        # Resulting: shape: [batch_size, max_len, embed_size]
        x = x.to(device)
        final_embedding = self.embedding(x)
        # final_embedding_gpu = final_embedding
        # final_embedding_gpu = final_embedding.to(device)
        # print(f"  final_embedding_gpu: {final_embedding_gpu.shape}, final_embedding_gpu dtype: {final_embedding_gpu.dtype}, final_embedding_gpu device: {final_embedding_gpu.get_device()}")

        # print('Content of final_embedding_gpu:', final_embedding_gpu)

        max_len, batch_size = x.shape
        # print(f"batch_size: {batch_size}, max_len: {max_len}")
        # print(f"(batch_size, max_len, self.hidden_size): ({batch_size}, {max_len}, {self.hidden_units})")

        # x_with_posn_embedding = self.position_embedding(x)
        # print(f"x_with_posn_embedding.shape: {x_with_posn_embedding.shape}")

        self.position_embedding = self.position_embedding[:max_len, :, :]

        src_mask = self.make_src_mask(x)
        # print(f"src_mask.shape: {src_mask.shape}")

        x = self.dropout_layer(final_embedding + self.position_embedding.expand_as(final_embedding))

        # x_with_posn_embedding_dropout = self.dropout_layer(x_with_posn_embedding)
        # print(f"x_with_posn_embedding_dropout.shape: {x_with_posn_embedding_dropout.shape}")



        transformer_output = self.transformer_encoder(src=x, src_key_padding_mask = src_mask)

        # print(f"transformer_output.shape: {transformer_output.shape}")

        output = transformer_output

        return output     


# ## Sanity Check: Transformer Encoder
# 
# The code below runs a sanity check for your `TransformerEncoder` class. The tests are similar to the hidden ones in Gradescope. However, note that passing the sanity check does <b>not</b> guarantee that you will pass the autograder; it is intended to help you debug.

# In[31]:


### DO NOT EDIT ###

if __name__=="__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Set random seed
    torch.manual_seed(42)
    # Create test inputs
    embedding_dim = [4, 8, 12]
    max_len = [10,20,30,40,50,60,70,80,90]
    num_layers = [1,1,1,2,2,2,3,3,3]
    nheads = [1, 1, 1, 1, 2, 2, 2, 4, 4]
    dimf = [50, 100, 150]
    params = []
    inputs = []
    i = 0
    for df in dimf:
        for ed in embedding_dim:
            inp = {}
            inp['src_vocab'] = src_vocab
            inp['embedding_dim'] = ed
            inp['num_heads'] = nheads[i]
            inp['dim_feedforward'] = df
            inp['num_layers'] = num_layers[i]
            inp['max_len_src'] = max_len[i]
            inp['device'] = device
            inputs.append(inp)
            i += 1
    # Test init
    expected_outputs = [51890, 103858, 155954, 53340, 106736, 160388, 55690, 111314, 167322]

    sanityCheckModel(inputs, TransformerEncoder, expected_outputs, "init", None)


# In[32]:


### DO NOT EDIT ###

if __name__=="__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Set random seed
    torch.manual_seed(42)
    # Test forward
    inputs = []
    batch_sizes = [1, 2]
    dimf = 100
    embedding_dims = [32,64,128]
    nheads = iter([1, 1, 2, 2, 4, 4])
    num_layers = iter([1,1,2,2,3,3])
    max_len = iter([10,20,30,40,50,60])
    for ed in embedding_dims:
        for b in batch_sizes:
            inp = {}
            inp['src_vocab'] = src_vocab
            inp['embedding_dim'] = ed
            inp['num_heads'] = next(nheads)
            inp['dim_feedforward'] = dimf
            inp['num_layers'] = next(num_layers)
            inp['max_len_src'] = next(max_len)
            inp['device'] = device
            inp["batch_size"] = b
            inputs.append(inp)
    # create sanity datasets
    sanity_dataset = MyData(src_tensor_train, trg_tensor_train)
    sanity_loader = torch.utils.data.DataLoader(sanity_dataset, batch_size=50, num_workers=2, drop_last=True, shuffle=True)
    expected_outputs = [torch.Size([1, 16, 32]), torch.Size([2, 16, 32]), torch.Size([1, 16, 64]), torch.Size([2, 16, 64]), torch.Size([1, 16, 128]), torch.Size([2, 16, 128])]

    sanityCheckModel(inputs, TransformerEncoder, expected_outputs, "forward", sanity_loader)


# ## <font color='red'>TODO:</font> Decoder Model [10 points]
# Now we implement a Decoder model. Unlike the RNN, you do not need to explicitly compute inter-attention with the encoder; you will use the nn.TransformerDecoder model, which takes care of this for you.
# 
# In this cell, you should implement the `__init(...)` and `forward(...)` functions, each of which is <b>5 points</b>.

# In[33]:


class TransformerDecoder(nn.Module):
    def __init__(self, trg_vocab, embedding_dim, num_heads, num_layers, dim_feedforward, max_len_trg, device):
        super(TransformerDecoder, self).__init__()
        self.device = device
        """
        Args:
            trg_vocab: Vocab_Lang, the target vocabulary
            embedding_dim: the dimension of the embedding (also the number of expected features for the input of the Transformer)
            num_heads: The number of attention heads
            num_layers: the number of Transformer Decoder layers
            dim_feedforward: the dimension of the feedforward network models in the Transformer
            max_len_trg: maximum length of the target sentences
            device: the working device (you may need to map your postional embedding to this device)
        """
        self.trg_vocab = trg_vocab # Do not change
        trg_vocab_size = len(trg_vocab)

        # Create positional embedding matrix
        self.position_embedding = create_positional_embedding(max_len_trg, embedding_dim).to(device)
        self.register_buffer('positional_embedding', self.position_embedding) # this informs the model that positional_embedding is not a learnable parameter

        ### TODO ###

        # Initialize embedding layer
        self.embedding = nn.Embedding(trg_vocab_size, embedding_dim).to(device)

        # Dropout layer
        self.dropout_layer = (nn.Dropout(0.5)).to(device)

        # Initialize a nn.TransformerDecoder model (you'll need to use embedding_dim,
        # num_layers, num_heads, & dim_feedforward here)
        decoder_layer = nn.TransformerDecoderLayer(d_model=embedding_dim, dim_feedforward= dim_feedforward, nhead=num_heads ).to(device)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers).to(device)
        # memory = torch.rand(10, 32, 512)
        # tgt = torch.rand(20, 32, 512)
        # out = transformer_decoder(tgt, memory)




        # Final fully connected layer
        self.fc_layer = nn.Linear(embedding_dim , trg_vocab_size).to(device)


    def generate_square_subsequent_mask(self, sz):
        """Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)).to(self.device)
        return mask

    def forward(self, dec_in, enc_out):
        """
        Args:
            dec_in: [sequence length, batch_size]
            enc_out: [max_len, batch_size, embed_dim]
        Returns:
            output: [sequence length, batch_size, trg_vocab_size]
        Pseudo-code:
        - Compute input word and positional embeddings in similar manner to encoder
        - Call generate_square_subsequent_mask() to compute a mask: this time,
          the mask is to prevent the decoder from attending to tokens in the "future".
          In other words, at time step i, the decoder should only attend to tokens
          1 to i-1.
        - Call the decoder, with tgt_mask = trg_mask
        - Run the output through the fully-connected layer and return it
        """
        output = None

        ### TODO ###
        ### TODO ###
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        enc_out = enc_out.to(device)
        # x_int64 = x.type(torch.int64)

        # print('Content of embedding:', x_int64)
        # print('Shape of embedding:', x_int64.shape, '\n')
        # print('Type of embedding:', x_int64.dtype, '\n')

        # Resulting: shape: [batch_size, max_len, embed_size]
        x = dec_in.to(device)
        x = x.type(torch.int64)
        final_embedding = self.embedding(x)
        # final_embedding_gpu = final_embedding
        # final_embedding_gpu = final_embedding.to(device)
        # print(f"  final_embedding_gpu: {final_embedding_gpu.shape}, final_embedding_gpu dtype: {final_embedding_gpu.dtype}, final_embedding_gpu device: {final_embedding_gpu.get_device()}")

        # print('Content of final_embedding_gpu:', final_embedding_gpu)

        max_len, batch_size = x.shape
        # print(f"batch_size: {batch_size}, max_len: {max_len}")
        # print(f"(batch_size, max_len, self.hidden_size): ({batch_size}, {max_len}, {self.hidden_units})")

        # x_with_posn_embedding = self.position_embedding(x)
        # print(f"x_with_posn_embedding.shape: {x_with_posn_embedding.shape}")

        self.position_embedding = self.position_embedding[:max_len, :, :]

        trg_mask = self.generate_square_subsequent_mask(dec_in.shape[0])
        # print(f"trg_mask.shape: {trg_mask.shape}")
        trg_mask = trg_mask.to(device)

        x = self.dropout_layer(final_embedding + self.position_embedding.expand_as(final_embedding))
        x = x.to(device)

        # x_with_posn_embedding_dropout = self.dropout_layer(x_with_posn_embedding)
        # print(f"x_with_posn_embedding_dropout.shape: {x_with_posn_embedding_dropout.shape}")



        transformer_output = self.transformer_decoder(tgt=x,memory=enc_out, tgt_mask = trg_mask)

        # print(f"transformer_output.shape: {transformer_output.shape}")



        output = self.fc_layer(transformer_output)

        return output    


# ## Sanity Check: Transformer Decoder
# 
# The code below runs a sanity check for your `TransformerDecoder` class. The tests are similar to the hidden ones in Gradescope. However, note that passing the sanity check does <b>not</b> guarantee that you will pass the autograder; it is intended to help you debug.

# In[34]:


### DO NOT EDIT ###

def sanityCheckTransformerDecoderModelForward(inputs, NN, expected_outputs):
    print('--- TEST: Output shape of forward(...) ---\n')
    msg = ''
    for i, inp in enumerate(inputs):
        input_rep = '{'
        for k,v in inp.items():
            if torch.is_tensor(v):
                input_rep += str(k) + ': ' + 'Tensor with shape ' + str(v.size()) + ', '
            else:
                input_rep += str(k) + ': ' + str(v) + ', '
        input_rep += '}'
        dec = NN(trg_vocab=inp['trg_vocab'],embedding_dim=inp['embedding_dim'],num_heads=inp['num_heads'],num_layers=inp['num_layers'],dim_feedforward=inp['dim_feedforward'],max_len_trg=inp['max_len_trg'],device=inp['device'])
        dec_in = torch.randint(low=0,high=20,size=(inp['max_len_trg'], inp['batch_size']))
        enc_out = torch.rand(inp['max_len_trg'], inp['batch_size'], inp['embedding_dim'])
        inp['encoder_outputs'] = enc_out
        with torch.no_grad(): 
            stu_out = dec(enc_out=enc_out, dec_in=dec_in)
        del dec
        has_passed = True
        if not torch.is_tensor(stu_out):
            has_passed = False
            msg = 'Output must be a torch.Tensor; received ' + str(type(stu_out))
        status = 'PASSED' if has_passed else 'FAILED'
        if not has_passed:
            message = '\t' + status + "\t Init Input: " + input_rep + '\tForward Input Shape: ' + str(inp['encoder_outputs'].shape) + '\tExpected Output Shape: ' + str(expected_outputs[i]) + '\t' + msg
            print(message)
            continue
        
        has_passed = stu_out.size() == expected_outputs[i]
        msg = 'Your Output Shape: ' + str(stu_out.size())
        status = 'PASSED' if has_passed else 'FAILED'
        message = '\t' + status + "\t Init Input: " + input_rep + '\tForward Input Shape: ' + str(inp['encoder_outputs'].shape) + '\tExpected Output Shape: ' + str(expected_outputs[i]) + '\t' + msg
        print(message)
        


# In[35]:


### DO NOT EDIT ###

if __name__ == '__main__':
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Set random seed
    torch.manual_seed(42)
    # Create test inputs
    hidden_units = [50, 100, 200]
    embedding_dim = [8, 16]
    num_heads = [1, 2]
    dim_feedforward = [50, 100]
    num_layers = [1, 2]
    max_lens = 64
    params = []
    inputs = []
    for ed in embedding_dim:
        for df in dim_feedforward:
            for nh in num_heads:
                for nl in num_layers:
                    inp = {}
                    inp['trg_vocab'] = trg_vocab
                    inp['embedding_dim'] = ed
                    inp['num_heads'] = nh
                    inp['num_layers'] = nl
                    inp['dim_feedforward'] = df
                    inp['max_len_trg'] = max_lens
                    inp['device'] = device
                    inputs.append(inp)
    # Test init
    expected_outputs = [113835, 115317, 113835, 115317, 114685, 117017, 114685, 117017]
    sanityCheckModel(inputs, TransformerDecoder, expected_outputs, "init", None)
    print()

    # Test forward
    inputs = []
    batch_sizes = [1, 2, 4]
    num_heads = 2
    num_layers = 1
    embedding_dims = iter([100, 100, 200, 200, 200, 400, 400, 800, 800])
    expected_outputs = [torch.Size([16, 1, 6609]),torch.Size([16, 2, 6609]),torch.Size([16, 4, 6609]),torch.Size([32, 1, 6609]),torch.Size([32, 2, 6609]),torch.Size([32, 4, 6609]),torch.Size([64, 1, 6609]),torch.Size([64, 2, 6609]),torch.Size([128, 4, 6609])]
    max_lens = iter([16, 16, 16, 32, 32, 32, 64, 64, 128])

    for hu in hidden_units:
        for b in batch_sizes:
            inp = {}
            edim = next(embedding_dims)
            inp['embedding_dim'] = edim
            inp['trg_vocab'] = trg_vocab
            inp['num_heads'] = num_heads
            inp['num_layers'] = num_layers
            inp["batch_size"] = b
            inp['dim_feedforward'] = hu
            inp['max_len_trg'] = next(max_lens)
            inp['device'] = device
            inputs.append(inp)
    
    sanityCheckTransformerDecoderModelForward(inputs, TransformerDecoder, expected_outputs)


# ## Train Transformer Model
# 
# Like the RNN, we train the encoder and decoder using cross-entropy loss.

# In[36]:


### DO NOT EDIT ###

def train_transformer_model(encoder, decoder, dataset, optimizer, device, n_epochs):
    encoder.train()
    decoder.train()
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    for epoch in range(n_epochs):
        start = time.time()
        losses = []

        for src, trg in tqdm(train_dataset):
            
            src = src.to(device).transpose(0,1) # [max_src_length, batch_size]
            trg = trg.to(device).transpose(0,1) # [max_trg_length, batch_size]

            enc_out = encoder(src)
            output = decoder(trg[:-1, :], enc_out)

            output = output.reshape(-1, output.shape[2])
            trg = trg[1:].reshape(-1)

            optimizer.zero_grad()

            loss = criterion(output, trg)
            losses.append(loss.item())

            loss.backward()

            # Clip to avoid exploding grading issues
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1)

            optimizer.step()

        mean_loss = sum(losses) / len(losses)
        print('Epoch:{:2d}/{}\t Loss:{:.4f} ({:.2f}s)'.format(epoch + 1, n_epochs, mean_loss, time.time() - start))


# In[37]:


### DO NOT EDIT ###

if __name__ == '__main__':
    # HYPERPARAMETERS - feel free to change
    LEARNING_RATE = 0.001
    DIM_FEEDFORWARD=512
    N_EPOCHS=1
    N_HEADS=2
    N_LAYERS=2

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transformer_encoder = TransformerEncoder(src_vocab, EMBEDDING_DIM, N_HEADS, 
                                 N_LAYERS,DIM_FEEDFORWARD,
                                 max_length_src, device).to(device)
    transformer_decoder = TransformerDecoder(trg_vocab, EMBEDDING_DIM, N_HEADS, 
                              N_LAYERS,DIM_FEEDFORWARD,
                              max_length_trg, device).to(device)

    transformer_model_params = list(transformer_encoder.parameters()) + list(transformer_decoder.parameters())
    optimizer = torch.optim.Adam(transformer_model_params, lr=LEARNING_RATE)

    print('Encoder and Decoder models initialized!')


# In[38]:


### DO NOT EDIT ###

if __name__ == '__main__':
    train_transformer_model(transformer_encoder, transformer_decoder, train_dataset, optimizer, device, N_EPOCHS)


# ## <font color='red'>TODO:</font> Inference (Decoding) Function [5 points]
# 
# Now that we have trained the model, we can use it on test data.
# 
# Here, you will write a function that takes your trained transformer model and a source sentence (Spanish), and returns its translation (English sentence). Like the RNN, we use the prediction of the decoder as the input to the decoder for the sequence of outputs. For the RNN, at time step $t_i$ the decoder takes the hidden state $h_{i-1}$ and the previous prediction $w_{i-1}$ at each time step. However, because the transformer does not use recurrences, we do not pass a hidden state; instead, at time step $t_i$ we pass $w_1,w_2 \cdots w_{i-1}$, which is the entire sequence predicted so far.

# In[39]:


def decode_transformer_model(encoder, decoder, src, max_decode_len, device):
    """
    Args:
        encoder: Your TransformerEncoder object
        decoder: Your TransformerDecoder object
        src: [max_src_length, batch_size] the source sentences you wish to translate
        max_decode_len: The maximum desired length (int) of your target translated sentences
        device: the device your torch tensors are on (you may need to call x.to(device) for some of your tensors)

    Returns:
        curr_output: [batch_size, max_decode_len] containing your predicted translated sentences
        curr_predictions: [batch_size, max_decode_len, trg_vocab_size] containing the (unnormalized) probabilities of each
            token in your vocabulary at each time step

    Pseudo-code:
    - Obtain encoder output by encoding src sentences
    - For 1 ≤ t ≤ max_decode_len:
        - Obtain dec_input as the best words so far for previous time steps (you can get this from curr_output)
        - Obtain your (unnormalized) prediction probabilities by feeding dec_input and encoder output to decoder
        - Save your (unnormalized) prediction probabilities in curr_predictions at index t
        - Calculate the most likely (highest probability) token and save in curr_output at timestep t
    """
    # Initialize variables
    trg_vocab = decoder.trg_vocab
    batch_size = src.size(1)
    curr_output = torch.zeros((batch_size, max_decode_len))
    curr_predictions = torch.zeros((batch_size, max_decode_len, len(trg_vocab.idx2word)))
    enc_output = None

    # We start the decoding with the start token for each example
    dec_input = torch.tensor([[trg_vocab.word2idx['<start>']]] * batch_size).transpose(0,1)
    curr_output[:, 0] = dec_input.squeeze(1)
    
    ### TODO: Implement decoding algorithm ###
    encoder_output = encoder(src)
    encoder_output = encoder_output.to(device)

    # print(f"max_decode_len: {max_decode_len}")

    for t in range(1,max_decode_len):
        # print(f"INSIDE FOR LOOP T is: {t}")
        # if t==0:
        #     x = dec_input.to(device)
        #     # print(f"x when t={t}: {x}, x.shape={x.shape}")
        # elif t>0:
        #     x_int = (dec_input[:, -1]).to(device)
        #     # print(f"x when t={t}: {x}, x.shape={x.shape}")
        #     x = x_int.view(batch_size,1).to(device)


        enc_output = encoder_output.to(device)
        dec_input = curr_output[ : , :t]
        decoder_output_unnormalized_pred_probs = decoder(dec_input, encoder_output)
        decoder_output = decoder_output_unnormalized_pred_probs.to(device)
        print('Shape of decoder_output  (Unnormalized) output distribution [batch_size, vocab_size]:', decoder_output.shape)



        curr_predictions[:, t, :] = decoder_output.to(device)
        # print('Shape of curr_predictions : [batch_size, max_decode_len, trg_vocab_size]:', curr_predictions.shape, '\n')


        new_dec_input = (torch.argmax(decoder_output, dim=1)).to(device)
        print('Shape of new_dec_input:', new_dec_input.shape, '\n')

        # new_dec_input_unsqueezed = (torch.argmax(fc_out, dim=1).unsqueeze(1)).to(device)
        # print('Shape of new_dec_input_unsqueezed:', new_dec_input_unsqueezed.shape, '\n')

        # print('Shape of dec_input before concatenation:', dec_input.shape, '\n')
        # dec_input = (torch.cat([dec_input, new_dec_input_unsqueezed], dim=1)).to(device)
        # print('Shape of dec_input after concatenation:', dec_input.shape, '\n')
        # print(f"dec_input: {dec_input}")

        # dec_input.append(new_dec_input)
        # dec_input = ((torch.argmax(fc_out, dim=1)).unsqueeze(1)).to(device)
        # print('new_dec_input:', new_dec_input)


        curr_output[:, t] = new_dec_input
        # print('curr_output:', curr_output)
        # print('Shape of curr_output: [batch_size, max_decode_len]:', curr_output.shape, '\n')

    #

    return curr_output, curr_predictions, enc_output


# You can run the cell below to qualitatively compare some of the sentences your model generates with the some of the correct translations.

# In[40]:


### DO NOT EDIT ###

if __name__ == '__main__':
    transformer_encoder.eval()
    transformer_decoder.eval()
    idxes = random.choices(range(len(test_dataset.dataset)), k=5)
    src, trg =  train_dataset.dataset[idxes]
    curr_output, _, _ = decode_transformer_model(transformer_encoder, transformer_decoder, src.transpose(0,1).to(device), trg.size(1), device)
    for i in range(len(src)):
        print("Source sentence:", ' '.join([x for x in [src_vocab.idx2word[j.item()] for j in src[i]] if x != '<pad>']))
        print("Target sentence:", ' '.join([x for x in [trg_vocab.idx2word[j.item()] for j in trg[i]] if x != '<pad>']))
        print("Predicted sentence:", ' '.join([x for x in [trg_vocab.idx2word[j.item()] for j in curr_output[i]] if x != '<pad>']))
        print("----------------")


# ## Evaluate Transformer Model [20 points]
# 
# Now we can run the test set through the transformer model. We expect your BLEU scores to satisfy the following conditions: 
# 
# *   BLEU-1 > 0.290
# *   BLEU-2 > 0.082
# *   BLEU-3 > 0.060
# *   BLEU-4 > 0.056
# 

# In[ ]:


### DO NOT EDIT ###

def evaluate_model(encoder, decoder, test_dataset, target_tensor_val, device):
    trg_vocab = decoder.trg_vocab
    batch_size = test_dataset.batch_size
    n_batch = 0
    total_loss = 0

    encoder.eval()
    decoder.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    losses=[]
    final_output, target_output = None, None

    with torch.no_grad():
        for batch, (src, trg) in enumerate(test_dataset):
            n_batch += 1
            loss = 0
            
            src, trg = src.transpose(0,1).to(device), trg.transpose(0,1).to(device)
            curr_output, curr_predictions, enc_out = decode_transformer_model(encoder, decoder, src, trg.size(0), device)

            for t in range(1, trg.size(0)):
                output = decoder(trg[:-1, :], enc_out)
                output = output.reshape(-1, output.shape[2])
                loss_trg = trg[1:].reshape(-1)
                loss += criterion(output, loss_trg)
                # loss += criterion(curr_predictions[:,t,:].to(device), trg[t,:].reshape(-1).to(device))

            if final_output is None:
                final_output = torch.zeros((len(target_tensor_val), trg.size(0)))
                target_output = torch.zeros((len(target_tensor_val), trg.size(0)))

            final_output[batch*batch_size:(batch+1)*batch_size] = curr_output
            target_output[batch*batch_size:(batch+1)*batch_size] = trg.transpose(0,1)
            losses.append(loss.item() / (trg.size(0)-1))

        mean_loss = sum(losses) / len(losses)
        print('Loss {:.4f}'.format(mean_loss))
    
    # Compute Bleu scores
    return compute_bleu_scores(target_tensor_val, target_output, final_output, trg_vocab)


# In[ ]:


### DO NOT EDIT ###

if __name__ == '__main__':
    transformer_save_candidate, transformer_scores = evaluate_model(transformer_encoder, transformer_decoder, test_dataset, trg_tensor_val, device)


# # What to Submit
# 
# To submit the assignment, download this notebook as a <TT>.py</TT> file. You can do this by going to <TT>File > Download > Download .py</TT>. Then rename it to `hwk3.py`.
# 
# You will also need to save the `rnn_encoder`, `rnn_decoder`, `transformer_encoder` and `transformer_decoder`. You can run the cell below to do this. After you save the files to your Google Drive, you need to manually download the files to your computer, and then submit them to the autograder.
# 
# You will submit the following files to the autograder:
# 1.   `hwk3.py`, the download of this notebook as a `.py` file (**not** a `.ipynb` file)
# 1.   `rnn_encoder.pt`, the saved version of your `rnn_encoder`
# 1.   `rnn_decoder.pt`, the saved version of your `rnn_decoder`
# 1.   `transformer_encoder.pt`, the saved version of your `transformer_encoder`
# 1.   `transformer_decoder.pt`, the saved version of your `transformer_decoder`

# In[ ]:


### DO NOT EDIT ###

import pickle


# In[ ]:


### DO NOT EDIT ###

if __name__=='__main__':
    # from google.colab import drive
    # drive.mount('/content/drive')
    print()
    if rnn_encoder is not None and rnn_encoder is not None:
        print("Saving RNN model....") 
        # torch.save(rnn_encoder, 'drive/My Drive/rnn_encoder.pt')
        # torch.save(rnn_decoder, 'drive/My Drive/rnn_decoder.pt')
        torch.save(rnn_encoder, 'saved_models/rnn_encoder.pt')
        torch.save(rnn_decoder, 'saved_models/rnn_decoder.pt')
    if transformer_encoder is not None and transformer_decoder is not None:
        print("Saving Transformer model....") 
        # torch.save(transformer_encoder, 'drive/My Drive/transformer_encoder.pt')
        # torch.save(transformer_decoder, 'drive/My Drive/transformer_decoder.pt')
        torch.save(transformer_encoder, 'saved_models/transformer_encoder.pt')
        torch.save(transformer_decoder, 'saved_models/transformer_decoder.pt')


# In[ ]:




