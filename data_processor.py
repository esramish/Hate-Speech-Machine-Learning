import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
STOPWORDS = stopwords.words('english')
import re

def process_file(filename, stop_after_rows=None):
    data = pd.read_csv(filename)
    data = data.values
    posts = data[:stop_after_rows,1]
    vectorizer = CountVectorizer()
    preprocessor = vectorizer.build_preprocessor()
    X = np.empty(0)
    Y = np.empty(0)
    print("Preprocessing progress (by rows of original data):")
    for i in range(posts.shape[0]):
        if i % 100 == 0: print("%.0f%%" % (i*100/posts.shape[0]))
        row_posts_string = preprocessor(posts[i]) # preprocess the posts in this row (including making them lowercase)
        row_posts_list = row_posts_string.split('\n')[:-1] # split up all the posts in a given row (and ignore the last one, since it's always empty)
        j = 1
        for post in row_posts_list:
            post = post[(post.index('.') + 1):].strip() # remove the prepended index (e.g. "2.") and tab characters
            # TODO: more preprocessing
                # remove stopwords
                # remove unique words (maybe this isn't the right place to do that? idk)
            # get rid of URLs
            post = re.sub( r'http\S+', '', post )
                # emojis -- not sure
                # address misspelling of significant words
            X = np.append(X, post) # add it to our 1D numpy array of all posts
            
            #Check if theres no response
            if type(data[i,2]) != float:
                #Remove brackets from idx entries
                temp = data[i,2].replace('[', '')
                temp = temp.replace(']', '')
            #If post matches hate_speech_idx, add 1 to Y
            if str(j) in temp:
                Y = np.append(Y, 1)
            else: 
                Y = np.append(Y, 0)
            j += 1
    
    counts = vectorizer.fit_transform(X) # counts in a 2D matrix
    print(np.array(vectorizer.get_feature_names())[np.nonzero(counts[0])[1]])
    print(counts[0]) # just for playing around/testing
    # etc.     

    # data = re.sub(r'http\S+','',data)
    
    

def main():
    gab_data = process_file('data/gab.csv', stop_after_rows=50)
    #reddit_data = process_file('data/reddit.csv')

if __name__ == "__main__":
    main()

