import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from gensim.models import Word2Vec
from matplotlib import pyplot
from random import sample
from collections import Counter
from sklearn.model_selection import train_test_split
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#=== read data
df = pd.read_csv ('cellphone_data_srhdist200.csv')
df = pd.read_csv ('Predict.csv')

df = df.dropna(subset=['Price'])
df['Price'] = df['Price'].str.replace(r'$', '')
df['Price'] = df['Price'].str.replace(r',', '')
df['Price']=pd.to_numeric(df['Price'])
df = df[df.Price < 851]
title=list()
for i in df['Title']:
    title.append(i)
description=list()
for i in df['Description']:
    description.append(i.replace("QR Code Link to This Post   ",""))#remove QR Code Link to This Post
price=list()
for i in df['Price']:
    price.append(i)
attributes=list()
attributes_replace = {'[': '',
                      ']': '',
                      ', ': ' ',
                      '\'': '',
                      'condition: ': '',
                      'make / manufacturer: ': '',
                      'mobile OS: ': '',
                      'model name / number: ': '',
                      'size / dimensions: ': ''
                      }
for i in df['Attributes']:
    for key, value in attributes_replace.items():
        # Replace key character with value character in string
        i = i.replace(key, value)
    attributes.append(i)
len(description)
#===price interval
price_label=list()
intervals=list()
for i in range(40):
    intervals.append(range((i*50)+1,((i+1)*50)+1))
for i in price:
    for j in range(len(intervals)):
        if i in intervals[j]:
            price_label.append(j)
sorted(Counter(price_label).items())
#=== add title description attributes together
title_desc_attri=list()
for i in range(len(title)):
    title_desc_attri.append(title[i]+' '+description[i]+' '+attributes[i])
#=== tokenize
tokenized = [nltk.word_tokenize(doc.lower()) for doc in title_desc_attri]
lemmatizer= nltk.stem.WordNetLemmatizer()
lemmatizer_review=list()
for rev in tokenized: #lemmatize all review word
    lemmatized_token = [lemmatizer.lemmatize(token.lower()) for token in rev] #also lowercase each word
    lemmatizer_review.append(lemmatized_token)

#==remove punct
detokenized_lemmatizer=[]
for i in lemmatizer_review: #detokenize the tokened stop_words_removed_review from step3 output
    temp=''
    for j in range(len(i)):
      if j != len(i)-1:
          temp= temp +  i[j] + ' '
      else:
          temp = temp + i[j]
    detokenized_lemmatizer.append(temp)
tokenizerR = RegexpTokenizer(r'\w+')
non_punctuation=list()
for i in detokenized_lemmatizer:
    non_punctuation.append(tokenizerR.tokenize(i))

#stopwords
stop_words_removed_review = list()
for rev in non_punctuation:
    stop_words_removed = [token for token in rev if not token in stopwords.words('english')]
    #stop_words_removed = [token for token in rev if not token in stopwords.words('english') if token.isalpha()]
    #remove all the stop-words and the punctuations
    stop_words_removed_review.append(stop_words_removed)

for i in stop_words_removed_review:
    for j in i:
    if '_' in j: i.remove(j)

#tdidf
detokenized_stopwords=[]
for i in stop_words_removed_review: #detokenize the tokened stop_words_removed_review from step3 output
    temp=''
    for j in range(len(i)):
      if j != len(i)-1:
          temp= temp +  i[j] + ' '
      else:
          temp = temp + i[j]
    detokenized_stopwords.append(temp)

vectorizer12gram3min_de = TfidfVectorizer(min_df=3)
vectorizer12gram3min_de.fit(detokenized_stopwords)
print(vectorizer12gram3min_de.vocabulary_)
tfidf_dict=vectorizer12gram3min_de.vocabulary_
vectorizer12gram3min_de_vec = vectorizer12gram3min_de.transform(detokenized_stopwords)
print(vectorizer12gram3min_de_vec.toarray())
tfidf_array=vectorizer12gram3min_de_vec.toarray()
#pd.DataFrame(tfidf_array).to_csv("TFIDF_num.csv")

#===LSTM
# define vocab
from collections import Counter
words = [j for i in stop_words_removed_review for j in i]
count_words = Counter(words)
total_words = len(words)
sorted_words = count_words.most_common(total_words)
vocab_to_int = {w: i+1 for i, (w,c) in enumerate(sorted_words)} # note 0 is for null

text_int = []
for i in stop_words_removed_review:
    r = [vocab_to_int[w] for w in i]
    text_int.append(r)

X_training, X_test, y_training, y_test= train_test_split(weight_avg_ls_bal, price_list_bal, test_size=0.10, random_state=1 )

# build model
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten
from keras.layers import LSTM
max_features = total_words
maxlen = 80
batch_size = 32

# padding
x_train = sequence.pad_sequences(X_training, maxlen=maxlen)
x_test = sequence.pad_sequences(X_test, maxlen=maxlen)

# model architecture
model = Sequential()
model.add(Embedding(max_features, 40, input_length=maxlen))
model.add(LSTM(100, dropout=0.20, recurrent_dropout=0.20))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_training, batch_size=batch_size, epochs=20, validation_data=(x_test, y_test))

#===word2vec
# train model
model = Word2Vec(stop_words_removed_review, vector_size=300, min_count=3, epochs=100, window=5) #size is the size of output vector
model.save("word2vec_num_price851_num.model")
model = Word2Vec.load("word2vec_num_price851_num.model")
model.wv['io']
v = model.wv[model.wv.key_to_index]
words = list(model.wv.key_to_index)
#print(words)
len(words)

#pyplot.scatter(v[:, 0], v[:, 1])
#for i, word in enumerate(words):
#    pyplot.annotate(word, xy=(v[i, 0], v[i, 1]))
#pyplot.show()
#sim_is = model.wv.most_similar('galaxy', topn=3)
#print(sim_is)

##pre-trained word2vec
from gensim.models import KeyedVectors
filename = 'GoogleNews-vectors-negative300.bin' # make sure the file is unzipped
modelw2v = KeyedVectors.load_word2vec_format(filename, binary=True)
#len(modelw2v['samsung'])
#modelw2v.vocab
#modelw2v.key_to_index

#pretrained weighted average
prew2v=list()
posts=list()
for i in range(len(stop_words_removed_review)):
    total_weight=0
    sum=0
    temp=list()
    for j in stop_words_removed_review[i]:
        if j in modelw2v.key_to_index:
            if j in tfidf_dict:
                temp.append(j)
                index=tfidf_dict[j]
                sum+=modelw2v[j]*tfidf_array[i][index]
                total_weight+=tfidf_array[i][index]
    prew2v.append(sum/total_weight)
    posts.append(temp)

#pretrained_sum
prew2v_sum=list()
posts=list()
for i in range(len(stop_words_removed_review)):
    sum=0
    temp=list()
    for j in stop_words_removed_review[i]:
        if j in modelw2v.key_to_index:
            temp.append(j)
            sum+=modelw2v[j]
    prew2v_sum.append(sum)
    posts.append(temp)

#===weighted average
weight_avg=list()
posts=list()
for i in range(len(stop_words_removed_review)):
    total_weight=0
    sum=0
    temp=list()
    for j in stop_words_removed_review[i]:
        if j in words:
            if j in tfidf_dict:
                temp.append(j)
                index=tfidf_dict[j]
                sum+=model.wv[j]*tfidf_array[i][index]
                total_weight+=tfidf_array[i][index]
    weight_avg.append(sum/total_weight)
    posts.append(temp)

#sum
onlysum=list()
posts=list()
for i in range(len(stop_words_removed_review)):
    sum=0
    temp=list()
    for j in stop_words_removed_review[i]:
        if j in words:
            temp.append(j)
            sum+=model.wv[j]
    onlysum.append(sum)
    posts.append(temp)
len(onlysum)
#for i in range
pd.DataFrame(posts).to_csv("posts_price851_num_sum_predict.csv")
#pd.DataFrame(weight_avg).to_csv("weight_avg.csv")
#pd.DataFrame(price_label).to_csv("price_label.csv")
prediction_phone=onlysum.copy()
#====readfile
#df_weight_avg = pd.read_csv ('weight_avg.csv')
#weight_avg_ls=list()

#for i in range(len(df_weight_avg.index)):
#    temp=list()
#    for j in range(300):
#        temp.append(df_weight_avg[str(j)][i])
#    weight_avg_ls.append(temp)
#df_price = pd.read_csv ('price_label.csv')
#price_list=list()
#for i in df_price['0']:
#    price_list.append(i)

#down sample
price_0=list()
for i in range(len(price_label)):
    if price_label[i] == 0:
       price_0.append(i)
#list(range(0,1479))

new_index = [new for new in list(range(0,1414)) if new not in price_0]
for i in sample(price_0,168):
    new_index.append(i)

weight_avg_ls_bal=list()
price_list_bal=list()
TFIDF_bal=list()
for i in new_index:
    weight_avg_ls_bal.append(onlysum[i])
    price_list_bal.append(price_label[i])
    TFIDF_bal.append(tfidf_array[i])
len(weight_avg_ls_bal)

#pd.DataFrame(weight_avg_ls_bal).to_csv("pretrain_DownSample.csv")
#pd.DataFrame(price_list_bal).to_csv("price_label_DownSample_num.csv")
#pd.DataFrame(TFIDF_bal).to_csv("TFIDF_bal_DownSample_num.csv")
weight_avg_ls_bal[1]
X_training, X_test, y_training, y_test= train_test_split(weight_avg_ls_bal, price_list_bal, test_size=0.10, random_state=1 )

#train_tfidf=tfidf_array[:1321]
#test_tfidf=tfidf_array[1321:1468]
#train_weight_avg_ls=weight_avg_ls[:1321]
#test_weight_avg_ls=weight_avg_ls[1321:1468]
#train_price_list=price_list[:1321]
#test_price_list=price_list[1321:1468]

##===Logit
Logitmodel = LogisticRegression(max_iter=10000)
#training
Logitmodel.fit(X_training, y_training)
price_pred_logit = Logitmodel.predict(X_test)
#evaluation
acc_logit = accuracy_score(y_test, price_pred_logit)
print("Logit model Accuracy:: {:.2f}%".format(acc_logit*100))
kkk=onlysum.copy()
onlysum=[]
onlysum.append(kkk[0])
onlysum[2]
len(onlysum)
#===RF DT
DTmodel = DecisionTreeClassifier()
RFmodel = RandomForestClassifier(n_estimators=50, max_depth=3, bootstrap=True, random_state=0) ## number of trees and number of layers/depth
 # training
DTmodel.fit(X_training, y_training)
y_pred_DT = DTmodel.predict(X_test)
RFmodel.fit(X_training, y_training)
y_pred_RF = RFmodel.predict(X_test)
acc_DT = accuracy_score(y_test, y_pred_DT)
print("Decision Tree Model Accuracy: {:.2f}%".format(acc_DT*100))
acc_RF = accuracy_score(y_test, y_pred_RF)
print("Random Forest Model Accuracy: {:.2f}%".format(acc_RF*100))

#==SVM
SVMmodel = LinearSVC(max_iter=10000)
# training
SVMmodel.fit(X_training, y_training)
y_pred_SVM = SVMmodel.predict(onlysum)
# evaluation
acc_SVM = accuracy_score(y_test, y_pred_SVM)
print("SVM model Accuracy: {:.2f}%".format(acc_SVM*100))

## Neural Network and Deep Learning
from sklearn.neural_network import MLPClassifier
DLmodel = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(3,2), random_state=1)
# training
DLmodel.fit(X_training, y_training)
y_pred_DL= DLmodel.predict(X_test)
# evaluation
acc_DL = accuracy_score(y_test, y_pred_DL)
print("DL model Accuracy: {:.2f}%".format(acc_DL*100))
