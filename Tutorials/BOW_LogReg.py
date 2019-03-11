import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict

torch.manual_seed(1)


data = [("me gusta comer en la cafeteria".split(), "SPANISH"),
        ("Give it to me".split(), "ENGLISH"),
        ("No creo que sea una buena idea".split(), "SPANISH"),
        ("No it is not a good idea to get lost at sea".split(), "ENGLISH"),
        ("Comment vous appelez-vous".split(), "FRENCH"),
        ("Quel temps fait-il".split(), "FRENCH")]

test_data = [("Yo creo que si".split(), "SPANISH"),
             ("it is lost on me".split(), "ENGLISH"),
             ("Qu'est-ce que vous faites".split(), "FRENCH")]

# Create mapping
word_to_ix = {}
for sample, _ in data + test_data:
    for token in sample:
        if token not in word_to_ix:
            word_to_ix[token] = len(word_to_ix)

# print(word_to_ix)

# Define labels as Numbers
label_to_ix = {"ENGLISH":1, "SPANISH":0, "FRENCH": 2}

len_vocab = len(word_to_ix)
num_labels = 3

class BoWClassifier(nn.Module):
    def __init__(self, num_labels, vocab_size):
        # has to be done always
        super(BoWClassifier, self).__init__()

        self.transformation = nn.Linear(vocab_size, num_labels)

    def forward(self, bow_vector):
        # Linearly map to the two labels
        x = self.transformation(bow_vector)
        # Apply softmax to get probability (to better learn using gradients)
        return F.log_softmax(x, dim=1)

def make_bow_vector(sentence, word_to_ix):
    vector = torch.zeros(len(word_to_ix))
    for token in sentence:
        vector[word_to_ix[token]] += 1

    return vector.view(1, -1)

def make_target(label, label_to_ix):
    return torch.LongTensor([label_to_ix[label]])

# print(make_target("SPANISH", label_to_ix))
model = BoWClassifier(num_labels, len_vocab)

loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)


# for param in model.parameters():
#     print(param)

# TRAINING
for epoch in range(100):
    for instance, label in data:
        # REMEMBER to clear out gradients for each instance
        model.zero_grad()

        # form vector
        vec = make_bow_vector(instance, word_to_ix)

        # Obtain the probabilities
        l_p = model(vec)

        # Computing Loss (LEARNING)
        target = make_target(label, label_to_ix)
        loss = loss_function(l_p, target)
        loss.backward()
        optimizer.step()

# Test corresponding param changes after training
# for param in model.parameters():
#     print(param)

def define_prediction(l, label_to_ix):
    max = -10000000
    index = 0
    m_id = 0
    for t in l:
        for val in t:
            if val > max:
                max = val
                m_id = index
            index += 1

    s = ""
    for k in label_to_ix:
        if label_to_ix[k] == m_id:
            s = k
    return s

# TESTING
with torch.no_grad():
    for instance, label in test_data:
        vec = make_bow_vector(instance, word_to_ix)
        l_p = model(vec)
        print(" ".join(instance), "\nPREDICTION: ", define_prediction(l_p, label_to_ix))
