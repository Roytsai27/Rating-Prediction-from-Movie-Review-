import pandas as pd
from sklearn.pipeline import Pipeline
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize  
from sklearn.feature_extraction.text import TfidfVectorizer ,TfidfTransformer,CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, KFold
import nltk
#Load data

df1 = pd.read_csv("training_data.csv")
df2 = pd.read_csv("test_data.csv")
text_train = df1["text"].values
text_test = df2["text"].values
y_train = df1["stars"].values

#prepocessing 

nltk.download('stopwords')
nltk.download('wordnet')

#build Tokenizer and Stemmer to handle the text data

class LemmaTokenizer(object):
	def __init__(self):
		self.wnl = WordNetLemmatizer()
	def __call__(self, doc):
		return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

stemmer = SnowballStemmer("english", ignore_stopwords=True)

#In order to trasform text data into vectors
#We combine the Stemmer with sklearn-Countvectorizer
class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])

#there's about 24000 words we want to reduce the dimension
# And Build a pipeline to make our code more clear
stemed_vec = StemmedCountVectorizer(min_df= 2,ngram_range=(1,2),tokenizer = LemmaTokenizer())

clf = Pipeline([("vec",stemed_vec),("tf-idf",TfidfTransformer()),
	            ("SVD",TruncatedSVD(n_components=3000)),("svr",SVR(C=10,gamma=1,verbose=1))])
#Use GridSearch to tune our parameters

param_grid = {'svr__C': [0.001, 0.01, 0.1, 1, 10], 'svr__gamma': [0.001, 0.01, 0.1, 1]
              'SVD__n_components':[1000,2000,3000,4000],'vec_ngram_range':[(1,1),(1,2),(1,3)],
              'tfidf__use_idf': (True, False)}
cv = KFold(shuffle=True)
grid = GridSearchCV(clf, param_grid=param_grid, cv=cv, verbose=3)
# grid.fit(X_train, y_train)
# grid.predict(X_test)
#print(grid.best_score_) # 0.4652
#print(grid.best_params_) # {'C': 10, 'gamma': 1}

#fit Model
clf.fit(text_train, y_train)
y_test = clf.predict(text_test)
print(y_test)

#Adjust the value to 1~5
for i in range(len(y_test)):
	if (y_test[i]>5) : y_test[i]=5
	if (y_test[i]<1) : y_test[i]=1

#create output
result=[]
for i in range(df2.shape[0]):
    result.append([df2["review_id"][i], y_test[i]])
result_df = pd.DataFrame(result)
result_df.to_csv("result_df19.csv", index = False, header = False)
