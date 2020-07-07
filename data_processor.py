import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
STOPWORDS = stopwords.words('english')
import re

def process_file(filename):
    data = pd.read_csv(filename)
    data = data.values
    posts = data[:,1]
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(posts) # counts in a 2D matrix
    preprocessor = vectorizer.build_preprocessor()
    for i in range(posts.shape[0]):
        comments = preprocessor(posts[i])
        comment_list = comments.split('\n')
        for j in range(len(comment_list[:-1])):
            comment = comment_list[i]
            comment_list[j] = comment[(comment.index('.') + 1):].strip()
        if i==36: print(comment_list)
    
    

    # data = re.sub(r'http\S+','',data)
    
    # Preprocessing thoughts: 
    # lowercase
    # get rid of URLs
    # emojis -- not sure
    # address misspelling of significant words

def main():
    gab_data = process_file('data/gab.csv')
    #gab_data = process_file('data/reddit.csv')

if __name__ == "__main__":
    main()

