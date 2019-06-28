from config import contractions, count_words
import numpy as np

train_article_list = []
train_title_list = []
with open('data/train/train.article.txt', 'r') as article, open('data/train/train.title.txt', 'r') as title:
    for content, summary in zip(article, title):         
            content = content.split()
            summary = summary.split()
            new_content = []
            new_summary = []
            for word  in content:
                if word in contractions:
                    new_content.append(contractions[word])
                else:
                    new_content.append(word)
            for word  in summary:
                if word in contractions:
                    new_summary.append(contractions[word])
                else:
                    new_summary.append(word)
                    
            content = " ".join(new_content)
            summary = " ".join(new_summary)
            
            if len(content)<1000:
                    train_article_list.append(content)
                    train_title_list.append(summary)  
            
print('Number of training article : ',len(train_article_list)) # 57153 in case of content having size less than 800, 79949 in case of aa the articles,61777 in case of 1000
print('Number of training title :', len(train_title_list))
#print(train_article_list[:5]) 

word_counts = {}

count_words(word_counts, train_article_list)
count_words(word_counts, train_title_list)
            
print('Size of Vocabulary :', len(word_counts))  

embeddings_index = {}
with open('glove/glove.42B.300d.txt', encoding='utf-8') as f:
    for line in f:
        values = line.split(' ')
        word = values[0]
        embedding = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = embedding
        

print('size of glove Word embeddings :', len(embeddings_index))

missing_words = 0
threshold = 10

for word, count in word_counts.items():
    if count > threshold:
        if word not in embeddings_index:
            missing_words += 1
            
missing_ratio = round(missing_words/len(word_counts),4)*100
            
print("Number of words missing from glove : ", missing_words)
print("Percent of words that are missing from vocabulary : {}%".format(missing_ratio))

vocab_to_int = {} 

value = 0
for word, count in word_counts.items():
    if count >= threshold or word in embeddings_index:
        vocab_to_int[word] = value
        value += 1

# Special tokens that will be added to our vocab
codes = ["<UNK>","<PAD>","<EOS>","<GO>"]   

# Add codes to vocab
for code in codes:
    vocab_to_int[code] = len(vocab_to_int)

# Dictionary to convert integers to words
int_to_vocab = {}
for word, value in vocab_to_int.items():
    int_to_vocab[value] = word

usage_ratio = round(len(vocab_to_int) / len(word_counts),4)*100


print("Total number of unique words : ", len(word_counts))
print("Number of words which occur more than 20 times in the vocabulary : ", len(vocab_to_int))
print("Percentage of words which occur more than 20 times in the vocabulary : {}%".format(usage_ratio))       
    