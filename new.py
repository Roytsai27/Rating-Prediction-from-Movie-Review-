import pandas as pd
from sklearn.pipeline import Pipeline
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize  
class LemmaTokenizer(object):
	def __init__(self):
		self.wnl = WordNetLemmatizer()
	def __call__(self, doc):
		return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]
df1 = pd.read_csv("training_data.csv")
df2 = pd.read_csv("test_data.csv")

text_train = df1["text"]
text_test = df2["text"]
y_train = df1["stars"]

from sklearn.feature_extraction.text import TfidfVectorizer ,TfidfTransformer,CountVectorizer
from sklearn.decomposition import TruncatedSVD

nltk.download('stopwords')
nltk.download('wordnet')
stemmer = SnowballStemmer("english", ignore_stopwords=True)
# stemmer = PorterStemmer()
class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])


svd = TruncatedSVD(n_components = 1000)
vectorizer = TfidfVectorizer()
vec = StemmedCountVectorizer(min_df= 2,ngram_range=(1,2),tokenizer = LemmaTokenizer())
# vectorizer.fit(text_train)

# X_train = vectorizer.transform(text_train)
# X_test = vectorizer.transform(text_test)

# X_SVD_train = svd.fit_transform(X_train)
# X_SVD_test = svd.transform(X_test)
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.linear_model import SGDRegressor


clf = Pipeline([("vec",vec),("tf-idf",TfidfTransformer()),
	            ("SVD",TruncatedSVD(n_components=3000)),("svc",SVR(C=10,gamma=1,verbose=1))])

#param_grid = {'C': [0.001, 0.01, 0.1, 1, 10], 'gamma': [0.001, 0.01, 0.1, 1]}
# param_grid = {"SVD__n_components":[1000,2000,3000]}
# cv = KFold(shuffle=True)
# clf = GridSearchCV(pipe, param_grid=param_grid, cv=cv, verbose=3)
#grid.fit(X_train, y_train)
#grid.predict(X_test)
#print(grid.best_score_) # 0.4652
#print(grid.best_params_) # {'C': 10, 'gamma': 1}
# clf = SVR(C=10, gamma=1)

clf.fit(text_train, y_train)
y_test = clf.predict(text_test)
print(y_test)
for i in range(len(y_test)):
	if (y_test[i]>5) : y_test[i]=5
	if (y_test[i]<1) : y_test[i]=1
result=[]
for i in range(df2.shape[0]):
    result.append([df2["review_id"][i], y_test[i]])
result_df = pd.DataFrame(result)
result_df.to_csv("result_df19.csv", index = False, header = False)
