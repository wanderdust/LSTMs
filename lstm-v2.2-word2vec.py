#!/usr/bin/env python
# coding: utf-8

# # Sentiment Analysis with an RNN
# 
# In this notebook, you'll implement a recurrent neural network that performs sentiment analysis. 
# >Using an RNN rather than a strictly feedforward network is more accurate since we can include information about the *sequence* of words. 
# 
# Here we'll use a dataset of movie reviews, accompanied by sentiment labels: positive or negative.
# 
# <img src="assets/reviews_ex.png" width=40%>
# 
# ### Network Architecture
# 
# The architecture for this network is shown below.
# 
# <img src="assets/network_diagram.png" width=40%>
# 
# >**First, we'll pass in words to an embedding layer.** We need an embedding layer because we have tens of thousands of words, so we'll need a more efficient representation for our input data than one-hot encoded vectors. You should have seen this before from the Word2Vec lesson. You can actually train an embedding with the Skip-gram Word2Vec model and use those embeddings as input, here. However, it's good enough to just have an embedding layer and let the network learn a different embedding table on its own. *In this case, the embedding layer is for dimensionality reduction, rather than for learning semantic representations.*
# 
# >**After input words are passed to an embedding layer, the new embeddings will be passed to LSTM cells.** The LSTM cells will add *recurrent* connections to the network and give us the ability to include information about the *sequence* of words in the movie review data. 
# 
# >**Finally, the LSTM outputs will go to a sigmoid output layer.** We're using a sigmoid function because positive and negative = 1 and 0, respectively, and a sigmoid will output predicted, sentiment values between 0-1. 
# 
# We don't care about the sigmoid outputs except for the **very last one**; we can ignore the rest. We'll calculate the loss by comparing the output at the last time step and the training label (pos or neg).

# In[13]:


"""
from gensim.models import Word2Vec


w2v_model = Word2Vec.load('trained_w2v.bin')
"""


# In[14]:



from gensim.models import KeyedVectors
filename = 'GoogleNews-vectors-negative300.bin'
w2v_model = KeyedVectors.load_word2vec_format(filename, binary=True)


# In[15]:


print(w2v_model.vectors)


# ---
# ### Load in and visualize the data

# In[16]:


import numpy as np

# read data from text files
with open('data/tweets_full.txt', 'r') as f:
    reviews = f.read()
with open('data/tweets_full_target.txt', 'r') as f:
    labels = f.read()


# In[17]:


# read data from text files
with open('data/tweets_authors.txt', 'r') as f:
    authors = f.read()


# In[18]:


print(reviews[:1000])
print()
print(labels[:20])
print()
print(authors[:20])


# ## Data pre-processing
# 
# The first step when building a neural network model is getting your data into the proper form to feed into the network. Since we're using embedding layers, we'll need to encode each word with an integer. We'll also want to clean it up a bit.
# 
# You can see an example of the reviews data above. Here are the processing steps, we'll want to take:
# >* We'll want to get rid of periods and extraneous punctuation.
# * Also, you might notice that the reviews are delimited with newline characters `\n`. To deal with those, I'm going to split the text into each review using `\n` as the delimiter. 
# * Then I can combined all the reviews back together into one big string.
# 
# First, let's remove all punctuation. Then get all the text without the newlines and split it into individual words.

# In[19]:


from string import punctuation

# get rid of punctuation
reviews = reviews.lower() # lowercase, standardize
all_text = ''.join([c for c in reviews if c not in punctuation])

# split by new lines and spaces
reviews_split = all_text.split('\n')
all_text = ' '.join(reviews_split)

# create a list of words
words = all_text.split()


# In[20]:


words[:39]


# ### Encoding the words
# 
# The embedding lookup requires that we pass in integers to our network. The easiest way to do this is to create dictionaries that map the words in the vocabulary to integers. Then we can convert each of our reviews into integers so they can be passed into the network.
# 
# > **Exercise:** Now you're going to encode the words with integers. Build a dictionary that maps words to integers. Later we're going to pad our input vectors with zeros, so make sure the integers **start at 1, not 0**.
# > Also, convert the reviews to integers and store the reviews in a new list called `reviews_ints`. 

# In[21]:


# feel free to use this import 
from collections import Counter

## Build a dictionary that maps words to integers
counts = Counter(words)
vocab = sorted(counts, key=counts.get, reverse=True)
vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}
vocab_to_int["none"] = 0
int_to_vocab = {v: k for k, v in vocab_to_int.items()}

## use the dict to tokenize each review in reviews_split
## store the tokenized reviews in reviews_ints
reviews_ints = []
for review in reviews_split:
    reviews_ints.append([vocab_to_int[word] for word in review.split()])


# **Test your code**
# 
# As a text that you've implemented the dictionary correctly, print out the number of unique words in your vocabulary and the contents of the first, tokenized review.

# In[22]:


# stats about vocabulary
print('Unique words: ', len((vocab_to_int)))  # should ~ 74000+
print()

# print tokens in first review
print('Tokenized review: \n', reviews_ints[:1])


# ### Encoding the labels
# 
# Our labels are "positive" or "negative". To use these labels in our network, we need to convert them to 0 and 1.
# 

# In[23]:


# 1=positive, 0=negative label conversion
labels_split = labels.split('\n')
labels_split = [label.strip() for label in labels_split]

word_2_int = {'neutral': 1, 'worry':0, 'happiness':0, 'sadness':0, 'love':1, 'surprise':1,
              'fun':1, 'relief':1, 'hate':0, 'empty':0, 
             'enthusiasm':1, 'boredom': 0, 'anger':0}

encoded_labels = np.array([0 if len(label) == 0 else word_2_int[label] for label in labels_split])


# In[24]:


print(encoded_labels)


# ### Encode the authors of each tweet

# **Remove punctuation**

# In[25]:


# get rid of punctuation
authors = authors.lower() # lowercase, standardize
all_authors_text = ''.join([c for c in authors if c not in punctuation])

# split by new lines and spaces
authors_split = all_authors_text.split('\n')
all_authors_text = ' '.join(authors_split)

# create a list of words
all_authors = all_authors_text.split()


# **Encoding the authors**

# In[26]:


# feel free to use this import 
from collections import Counter

## Build a dictionary that maps words to integers
counts_auth = Counter(all_authors)
vocab_auth = sorted(counts_auth, key=counts_auth.get, reverse=True)
vocab_to_int_auth = {word: ii for ii, word in enumerate(vocab_auth, 1)}

## use the dict to tokenize each review in reviews_split
## store the tokenized reviews in reviews_ints
authors_split = [author.strip() for author in authors_split][:-1]
authors_ints = [[vocab_to_int_auth[word] for word in authors_split]]

authors_ints = np.array(authors_ints).squeeze()


# ### Removing Outliers
# 
# As an additional pre-processing step, we want to make sure that our reviews are in good shape for standard processing. That is, our network will expect a standard input text size, and so, we'll want to shape our reviews into a specific length. We'll approach this task in two main steps:
# 
# 1. Getting rid of extremely long or short reviews; the outliers
# 2. Padding/truncating the remaining data so that we have reviews of the same length.
# 
# Before we pad our review text, we should check for reviews of extremely short or long lengths; outliers that may mess with our training.

# In[27]:


# outlier review stats
review_lens = Counter([len(x) for x in reviews_ints])
print("Zero-length reviews: {}".format(review_lens[0]))
print("Maximum review length: {}".format(max(review_lens)))


# Okay, a couple issues here. We seem to have one review with zero length. And, the maximum review length is way too many steps for our RNN. We'll have to remove any super short reviews and truncate super long reviews. This removes outliers and should allow our model to train more efficiently.
# 
# > **Exercise:** First, remove *any* reviews with zero length from the `reviews_ints` list and their corresponding label in `encoded_labels`.

# In[28]:


print('Number of reviews before removing outliers: ', len(reviews_ints))

## remove any reviews/labels with zero length from the reviews_ints list.

# get indices of any reviews with length 0
non_zero_idx = [ii for ii, review in enumerate(reviews_ints) if len(review) != 0]

# remove 0-length reviews and their labels
reviews_ints = [reviews_ints[ii] for ii in non_zero_idx]
encoded_labels = np.array([encoded_labels[ii] for ii in non_zero_idx])

print('Number of reviews after removing outliers: ', len(reviews_ints))


# ---
# ## Padding sequences
# 
# To deal with both short and very long reviews, we'll pad or truncate all our reviews to a specific length. For reviews shorter than some `seq_length`, we'll pad with 0s. For reviews longer than `seq_length`, we can truncate them to the first `seq_length` words. A good `seq_length`, in this case, is 200.
# 
# > **Exercise:** Define a function that returns an array `features` that contains the padded data, of a standard size, that we'll pass to the network. 
# * The data should come from `review_ints`, since we want to feed integers to the network. 
# * Each row should be `seq_length` elements long. 
# * For reviews shorter than `seq_length` words, **left pad** with 0s. That is, if the review is `['best', 'movie', 'ever']`, `[117, 18, 128]` as integers, the row will look like `[0, 0, 0, ..., 0, 117, 18, 128]`. 
# * For reviews longer than `seq_length`, use only the first `seq_length` words as the feature vector.
# 
# As a small example, if the `seq_length=10` and an input review is: 
# ```
# [117, 18, 128]
# ```
# The resultant, padded sequence should be: 
# 
# ```
# [0, 0, 0, 0, 0, 0, 0, 117, 18, 128]
# ```
# 
# **Your final `features` array should be a 2D array, with as many rows as there are reviews, and as many columns as the specified `seq_length`.**
# 
# This isn't trivial and there are a bunch of ways to do this. But, if you're going to be building your own deep learning networks, you're going to have to get used to preparing your data.

# In[29]:


def pad_features(reviews_ints, seq_length):
    ''' Return features of review_ints, where each review is padded with 0's 
        or truncated to the input seq_length.
    '''
    
    # getting the correct rows x cols shape
    features = np.zeros((len(reviews_ints), seq_length), dtype=int)

    # for each review, I grab that review and 
    for i, row in enumerate(reviews_ints):
        features[i, -len(row):] = np.array(row)[:seq_length]
    
    return features


# In[30]:


# Test your implementation!

seq_length = 30

features = pad_features(reviews_ints, seq_length=seq_length)

## test statements - do not change - ##
assert len(features)==len(reviews_ints), "Your features should have as many rows as reviews."
assert len(features[0])==seq_length, "Each feature row should contain seq_length values."

# print first 10 values of the first 30 batches 
print(features[:30,:10])


# ## Training, Validation, Test
# 
# With our data in nice shape, we'll split it into training, validation, and test sets.
# 
# > **Exercise:** Create the training, validation, and test sets. 
# * You'll need to create sets for the features and the labels, `train_x` and `train_y`, for example. 
# * Define a split fraction, `split_frac` as the fraction of data to **keep** in the training set. Usually this is set to 0.8 or 0.9. 
# * Whatever data is left will be split in half to create the validation and *testing* data.

# In[31]:


split_frac = 0.8

## split data into training, validation, and test data (features and labels, x and y)

split_idx = int(len(features)*split_frac)
train_x, remaining_x = features[:split_idx], features[split_idx:]
train_y, remaining_y = encoded_labels[:split_idx], encoded_labels[split_idx:]

test_idx = int(len(remaining_x)*0.5)
val_x, test_x = remaining_x[:test_idx], remaining_x[test_idx:]
val_y, test_y = remaining_y[:test_idx], remaining_y[test_idx:]

## print out the shapes of your resultant feature data
print("\t\t\tFeature Shapes:")
print("Train set: \t\t{}".format(train_x.shape), 
      "\nValidation set: \t{}".format(val_x.shape),
      "\nTest set: \t\t{}".format(test_x.shape))


# **Split the authors accordingly**

# In[32]:


authors_x, remaining_authors = authors_ints[:split_idx], authors_ints[split_idx:]

authors_val, authors_test = remaining_authors[:test_idx], remaining_authors[test_idx:]

print("\t\t\tFeature Shapes:")
print("Train set: \t\t{}".format(authors_x.shape), 
      "\nValidation set: \t{}".format(authors_val.shape),
      "\nTest set: \t\t{}".format(authors_test.shape))


# **Check your work**
# 
# With train, validation, and test fractions equal to 0.8, 0.1, 0.1, respectively, the final, feature data shapes should look like:
# ```
#                     Feature Shapes:
# Train set: 		 (train_size, word_length) 
# Validation set: 	(val_size, word_length) 
# Test set: 		  (test_size, word_length)
# ```

# ---
# ## DataLoaders and Batching
# 
# After creating training, test, and validation data, we can create DataLoaders for this data by following two steps:
# 1. Create a known format for accessing our data, using [TensorDataset](https://pytorch.org/docs/stable/data.html#) which takes in an input set of data and a target set of data with the same first dimension, and creates a dataset.
# 2. Create DataLoaders and batch our training, validation, and test Tensor datasets.
# 
# ```
# train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
# train_loader = DataLoader(train_data, batch_size=batch_size)
# ```
# 
# This is an alternative to creating a generator function for batching our data into full batches.

# In[ ]:





# In[33]:


import torch
from torch.utils.data import TensorDataset, DataLoader

# create Tensor datasets
train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y), torch.from_numpy(authors_x))
valid_data = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y), torch.from_numpy(authors_val))
test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y), torch.from_numpy(authors_test))

# dataloaders
batch_size = 50

# make sure the SHUFFLE your training data
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)


# In[34]:


# obtain one batch of training data
dataiter = iter(train_loader)
sample_x, sample_y, author = dataiter.next()

print('Sample input size: ', sample_x.size()) # batch_size, seq_length
print('Sample input: \n', sample_x)
print()
print('Sample label size: ', sample_y.size()) # batch_size
print('Sample label: \n', sample_y)
print()
print('Sample label size: ', author.size()) # batch_size
print('Sample label: \n', author)


# ---
# # Sentiment Network with PyTorch
# 
# Below is where you'll define the network.
# 
# <img src="assets/network_diagram.png" width=40%>
# 
# The layers are as follows:
# 1. An [embedding layer](https://pytorch.org/docs/stable/nn.html#embedding) that converts our word tokens (integers) into embeddings of a specific size.
# 2. An [LSTM layer](https://pytorch.org/docs/stable/nn.html#lstm) defined by a hidden_state size and number of layers
# 3. A fully-connected output layer that maps the LSTM layer outputs to a desired output_size
# 4. A sigmoid activation layer which turns all outputs into a value 0-1; return **only the last sigmoid output** as the output of this network.
# 
# ### The Embedding Layer
# 
# We need to add an [embedding layer](https://pytorch.org/docs/stable/nn.html#embedding) because there are 74000+ words in our vocabulary. It is massively inefficient to one-hot encode that many classes. So, instead of one-hot encoding, we can have an embedding layer and use that layer as a lookup table. You could train an embedding layer using Word2Vec, then load it here. But, it's fine to just make a new layer, using it for only dimensionality reduction, and let the network learn the weights.
# 
# 
# ### The LSTM Layer(s)
# 
# We'll create an [LSTM](https://pytorch.org/docs/stable/nn.html#lstm) to use in our recurrent network, which takes in an input_size, a hidden_dim, a number of layers, a dropout probability (for dropout between multiple layers), and a batch_first parameter.
# 
# Most of the time, you're network will have better performance with more layers; between 2-3. Adding more layers allows the network to learn really complex relationships. 
# 
# > **Exercise:** Complete the `__init__`, `forward`, and `init_hidden` functions for the SentimentRNN model class.
# 
# Note: `init_hidden` should initialize the hidden and cell state of an lstm layer to all zeros, and move those state to GPU, if available.

# In[35]:


# First checking if GPU is available
train_on_gpu=torch.cuda.is_available()

if(train_on_gpu):
    print('Training on GPU.')
else:
    print('No GPU available, training on CPU.')


# In[36]:



def int_2_word2vec(tensor, embed_size = 300):
    #1. Convert ints to words
    tensor = tensor.cpu().numpy()
    new_tensor_array = np.zeros(tensor.shape, dtype=object)
    
    for i, row in enumerate(tensor):
        for j, word in enumerate(row):
            new_tensor_array[i,j] = int_to_vocab[word]
            
    batch_size, length = tensor.shape
            
    #2. Pass words to wor2vec
    new_tensor = np.zeros((batch_size, length, embed_size), dtype=float)

    for i, batch in enumerate(new_tensor):
        for j, row in enumerate(batch):
            try:
                new_tensor[i,j] = w2v_model.wv[new_tensor_array[i,j]]
            except:
                new_tensor[i,j] = w2v_model.wv["none"]
            
    if train_on_gpu:
            return torch.tensor(new_tensor, dtype=torch.float64).cuda()

    else:
            return torch.tensor(new_tensor, dtype=torch.float64).cpu()

    


# In[37]:


import torch.nn as nn

class SentimentRNN(nn.Module):
    """
    The RNN model that will be used to perform Sentiment analysis.
    """

    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):
        """
        Initialize the model by setting up the layers.
        """
        super(SentimentRNN, self).__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        
        # embedding and LSTM layers
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(w2v_model.vectors))
        self.test = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_author = nn.Embedding(len(vocab_to_int_auth)+1, hidden_dim) # make it same dimension as h_t
            

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, 
                            dropout=drop_prob, batch_first=True)
        
        # dropout layer
        self.dropout = nn.Dropout(0.3)
        
        # linear and sigmoid layers
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sig = nn.Sigmoid()
        

    def forward(self, x, hidden, author):
        """
        Perform a forward pass of our model on some input and hidden state.
        """
        # embeddings and lstm_out
        x = x.long()
        
        # Embedding for tweet and auhors
        embeds = self.embedding(x)
        embeds_auth = self.embedding_author(author)
        
        # Lstm output
        lstm_out, hidden = self.lstm(embeds, hidden) # shape (batch_size, word_length, embedding_size)
        
        # Get the vector for last batch shape = (batch_size, embedding_size)
        lstm_out_last_batch = lstm_out[:, -1] # We are only getting the last word representation for each batch
        
        # ADD embedding and embedding_author tensors.
        fc_input = lstm_out_last_batch.add(embeds_auth)
                
        # dropout and fully-connected layer
        out = self.dropout(fc_input)
        out = self.fc(out)
        
        # sigmoid function
        sig_out = self.sig(out)
        # get last batch of labels
        sig_out = sig_out[:, -1].squeeze()
 
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
        


# ## Instantiate the network
# 
# Here, we'll instantiate the network. First up, defining the hyperparameters.
# 
# * `vocab_size`: Size of our vocabulary or the range of values for our input, word tokens.
# * `output_size`: Size of our desired output; the number of class scores we want to output (pos/neg).
# * `embedding_dim`: Number of columns in the embedding lookup table; size of our embeddings.
# * `hidden_dim`: Number of units in the hidden layers of our LSTM cells. Usually larger is better performance wise. Common values are 128, 256, 512, etc.
# * `n_layers`: Number of LSTM layers in the network. Typically between 1-3
# 
# > **Exercise:** Define the model  hyperparameters.
# 

# In[38]:


# Instantiate the model w/ hyperparams
vocab_size = len(vocab_to_int)+1 # +1 for the 0 padding + our word tokens
output_size = 1
embedding_dim = 300
hidden_dim = 256
n_layers = 2

net = SentimentRNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)

print(net)


# ---
# ## Training
# 
# Below is the typical training code. If you want to do this yourself, feel free to delete all this code and implement it yourself. You can also add code to save a model by name.
# 
# >We'll also be using a new kind of cross entropy loss, which is designed to work with a single Sigmoid output. [BCELoss](https://pytorch.org/docs/stable/nn.html#bceloss), or **Binary Cross Entropy Loss**, applies cross entropy loss to a single value between 0 and 1.
# 
# We also have some data and training hyparameters:
# 
# * `lr`: Learning rate for our optimizer.
# * `epochs`: Number of times to iterate through the training dataset.
# * `clip`: The maximum gradient value to clip at (to prevent exploding gradients).

# In[39]:


# loss and optimization functions
lr=0.001

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)


# In[ ]:


# training params
from tqdm import tqdm


epochs = 3 # 3-4 is approx where I noticed the validation loss stop decreasing

counter = 0
print_every = 100
clip=5 # gradient clipping
net.double()
# move model to GPU, if available
if(train_on_gpu):
    net.cuda()

net.train()
# train for some number of epochs
for e in tqdm(range(epochs)):
    # initialize hidden state
    h = net.init_hidden(batch_size)

    # batch loop
    for inputs, labels, authors in train_loader:
        counter += 1

        if(train_on_gpu):
            inputs, labels, authors = inputs.cuda(), labels.cuda(), authors.cuda()

        # Creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history
        h = tuple([each.data for each in h])

        # zero accumulated gradients
        net.zero_grad()

        # get the output from the model
        output, h = net(inputs, h, authors)

        # calculate the loss and perform backprop
        loss = criterion(output.squeeze(), labels.double())
        loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        nn.utils.clip_grad_norm_(net.parameters(), clip)
        optimizer.step()

        # loss stats
        if counter % print_every == 0:
            # Get validation loss
            val_h = net.init_hidden(batch_size)
            val_losses = []
            net.eval()
            for inputs, labels, authors in valid_loader:

                # Creating new variables for the hidden state, otherwise
                # we'd backprop through the entire training history
                val_h = tuple([each.data for each in val_h])

                if(train_on_gpu):
                    inputs, labels, authors = inputs.cuda(), labels.cuda(), authors.cuda()

                output, val_h = net(inputs, val_h, authors)
                val_loss = criterion(output.squeeze(), labels.double())

                val_losses.append(val_loss.item())

            net.train()
            print("Epoch: {}/{}...".format(e+1, epochs),
                  "Step: {}...".format(counter),
                  "Loss: {:.6f}...".format(loss.item()),
                  "Val Loss: {:.6f}".format(np.mean(val_losses)))


# ---
# ## Testing
# 
# There are a few ways to test your network.
# 
# * **Test data performance:** First, we'll see how our trained model performs on all of our defined test_data, above. We'll calculate the average loss and accuracy over the test data.
# 
# * **Inference on user-generated data:** Second, we'll see if we can input just one example review at a time (without a label), and see what the trained model predicts. Looking at new, user input data like this, and predicting an output label, is called **inference**.

# In[ ]:


# Get test data loss and accuracy

test_losses = [] # track loss
num_correct = 0

# init hidden state
h = net.init_hidden(batch_size)

net.eval()
# iterate over test data
for inputs, labels, authors in test_loader:

    # Creating new variables for the hidden state, otherwise
    # we'd backprop through the entire training history
    h = tuple([each.data for each in h])

    if(train_on_gpu):
        inputs, labels, authors = inputs.cuda(), labels.cuda(), authors.cuda()
    
    # get predicted outputs
    output, h = net(inputs, h, authors)
    
    # calculate loss
    test_loss = criterion(output.squeeze(), labels.float())
    test_losses.append(test_loss.item())
    
    # convert output probabilities to predicted class (0 or 1)
    pred = torch.round(output.squeeze())  # rounds to the nearest integer
    
    # compare predictions to true label
    correct_tensor = pred.eq(labels.float().view_as(pred))
    correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
    num_correct += np.sum(correct)


# -- stats! -- ##
# avg test loss
print("Test loss: {:.3f}".format(np.mean(test_losses)))

# accuracy over all test data
test_acc = num_correct/len(test_loader.dataset)
print("Test accuracy: {:.3f}".format(test_acc))


# ### Inference on a test review
# 
# You can change this test_review to any text that you want. Read it and think: is it pos or neg? Then see if your model predicts correctly!
#     
# > **Exercise:** Write a `predict` function that takes in a trained net, a plain text_review, and a sequence length, and prints out a custom statement for a positive or negative review!
# * You can use any functions that you've already defined or define any helper functions you want to complete `predict`, but it should just take in a trained net, a text review, and a sequence length.
# 

# In[ ]:


# negative test review
test_review_neg = 'The worst movie I have seen; acting was terrible and I want my money back. This movie had bad acting and the dialogue was slow.'


# In[ ]:


from string import punctuation

def tokenize_review(test_review):
    test_review = test_review.lower() # lowercase
    # get rid of punctuation
    test_text = ''.join([c for c in test_review if c not in punctuation])

    # splitting by spaces
    test_words = test_text.split()

    # tokens
    test_ints = []
    test_ints.append([vocab_to_int[word] for word in test_words])

    return test_ints

# test code and generate tokenized review
test_ints = tokenize_review(test_review_neg)
print(test_ints)


# In[ ]:


# test sequence padding
seq_length=35
features = pad_features(test_ints, seq_length)

print(features)


# In[ ]:


# test conversion to tensor and pass into your model
feature_tensor = torch.from_numpy(features)
print(feature_tensor.size())


# In[ ]:


def predict(net, test_review, author, sequence_length=200):
    
    net.eval()
    
    # tokenize review
    test_ints = tokenize_review(test_review)
    test_auth = vocab_to_int_auth[author]
    print(test_auth)
    
    # pad tokenized sequence
    seq_length=sequence_length
    features = pad_features(test_ints, seq_length)
    
    # convert to tensor to pass into your model
    feature_tensor = torch.from_numpy(features)
    author_tensor = torch.from_numpy(test_auth)
    
    batch_size = feature_tensor.size(0)
    
    # initialize hidden state
    h = net.init_hidden(batch_size)
    
    if(train_on_gpu):
        feature_tensor = feature_tensor.cuda()
        author_tensor = author_tensor.cuda()
    
    # get the output from the model
    output, h = net(feature_tensor, h, author_tensor)
    
    # convert output probabilities to predicted class (0 or 1)
    pred = torch.round(output.squeeze()) 
    # printing output value, before rounding
    print('Prediction value, pre-rounding: {:.6f}'.format(output.item()))
    
    # print custom response
    if(pred.item()==1):
        print("Positive")
    else:
        print("Negative")
        


# In[ ]:


# positive test review
test_review_pos = "I love my life!"
author = "tiffanylue"


# In[ ]:


# call function
seq_length=35 # good to use the length that was trained on

predict(net, test_review_pos, author, seq_length)


# ### Try out test_reviews of your own!
# 
# Now that you have a trained model and a predict function, you can pass in _any_ kind of text and this model will predict whether the text has a positive or negative sentiment. Push this model to its limits and try to find what words it associates with positive or negative.
# 
# Later, you'll learn how to deploy a model like this to a production environment so that it can respond to any kind of user data put into a web app!

# In[ ]:




