import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
STOPWORDS = stopwords.words('english')
import re

class Processor:

    def process_files(self, *filenames, stop_after_rows=None):
        '''Preprocess the post and label data from the given files. 
        If stop_after_rows is given, this process stops after that many file rows (even if not all of the files are reached, as such).'''
        data = pd.read_csv(filenames[0]).values
        for filename in filenames[1:]: 
            data = np.append(data, pd.read_csv(filename).values, axis=0)
        posts = data[:stop_after_rows,1]
        vectorizer = CountVectorizer()
        preprocessor = vectorizer.build_preprocessor()

        list_of_all_posts = np.empty(0)
        Y = np.empty(0)
        print("Preprocessing progress (by rows of original data):")
        for i in range(posts.shape[0]):
            if i % 100 == 0: print("%.0f%%" % (i*100/posts.shape[0]))
            row_posts_string = preprocessor(posts[i]) # preprocess the posts in this row (including making them lowercase)
            row_posts_list = row_posts_string.split('\n')[:-1] # split up all the posts in a given row (and ignore the last one, since it's always empty)
            j = 1
            for post in row_posts_list:
                post = post[(post.index('.') + 1):].strip() # remove the prepended index (e.g. "2.") and tab characters
                post = re.sub(r'\.|,|;|:|\?|!|\(|\)|\'|"|\u201C|\u201D', '', post) # remove certain punctuation
                
                # remove stopwords 
                post = re.sub(r'\u2018|\u2019', "'", post) # replace smart (curly) apostrophes with ASCII apostrophes, since that's what nltk uses
                post_words = post.split()
                post_words = list(filter(lambda word: word not in STOPWORDS, post_words))
                post = " ".join(post_words)
                
                # get rid of URLs
                post = re.sub( r'http\S+', '', post )
                
                # TODO: potential further preprocessing ideas:
                    # emojis -- not sure, might want to leave them (although we've already gotten rid of some punctuation and therefore punctuation-emojis, currently)
                    # address misspelling of significant words
                list_of_all_posts = np.append(list_of_all_posts, post) # add it to our 1D numpy array of all posts
                
                # Check if theres no response
                if type(data[i,2]) != float: # it's a string representation of a list
                    # Remove brackets from idx entries
                    temp = data[i,2].replace('[', '')
                    temp = temp.replace(']', '')
                    # Convert the string representation to an actual list of ints
                    temp_arr = list(map(lambda a: int(a), temp.split(',')))
                    #If post matches hate_speech_idx, add 1 to Y
                    if j in temp_arr:
                        Y = np.append(Y, 1)
                    else: 
                        Y = np.append(Y, 0)
                else: # it's 'n/a', which gets parsed as nan apparently. So none of these posts are marked as hate
                    Y = np.append(Y, 0)
                j += 1
        print("100%")

        counts = vectorizer.fit_transform(list_of_all_posts) # counts in a 2D matrix
        counts_np = np.array(counts.toarray()) # convert to normal numpy format

        feature_names = vectorizer.get_feature_names() # the 1D python list of features (i.e. words) that correspond to the columns of counts_np
        feature_names_np = np.array(feature_names) # convert to numpy

        # remove unique features/columns (i.e. words that appear only in one post throughout the corpus)
        non_unique_indeces = np.nonzero(np.count_nonzero(counts_np,axis=0)>1)[0] # the column indeces of the features that appear in more than one document throughout the corpus
        non_unique_counts_np = counts_np[:,non_unique_indeces] # select only the columns at those indeces
        non_unique_feature_names_np = feature_names_np[non_unique_indeces] # select only the feature names at those indeces

        return non_unique_counts_np, non_unique_feature_names_np, Y

        # print(np.array(vectorizer.get_feature_names())[np.nonzero(counts[0])[1]]) # good for seeing the word counts of a single post
    

def main():
    p = Processor()
    gab_X, gab_feature_names, gab_Y = p.process_files('data/gab.csv', stop_after_rows=50)
    # total_counts = gab_X.sum(0)

if __name__ == "__main__":
    main()

