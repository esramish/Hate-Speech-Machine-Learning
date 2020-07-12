import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
STOPWORDS = stopwords.words('english')
import re
import random
import ast

RANDOM_SEED = 13579

class Processor:

    def process_files(self, *filenames, stop_after_rows=None):
        random.seed(RANDOM_SEED)
        '''Preprocess the post and label data from the given files. 
        If stop_after_rows is given, this process stops after that many file rows (even if not all of the files are reached, as such).'''
        data = pd.read_csv(filenames[0]).values
        for filename in filenames[1:]: 
            data = np.append(data, pd.read_csv(filename).values, axis=0)
        posts = data[:stop_after_rows,1]
        r = data[:stop_after_rows,3]
     
        
        responses = []
        # print(responses[0])
        vectorizer = CountVectorizer()
        char_vectorizer = CountVectorizer(analyzer='char')
        preprocessor = vectorizer.build_preprocessor()

        list_of_all_posts = np.empty(0)
        Y = np.empty(0)
        print("Preprocessing progress (by rows of original data):")
        for i in range(posts.shape[0]):
            if i % 100 == 0: print("%.0f%%" % (i*100/posts.shape[0]))
            row_posts_string = preprocessor(posts[i]) # preprocess the posts in this row (including making them lowercase)
            row_posts_list = re.split(r'\n\d+\.', row_posts_string) # split up all the posts in a given row
            j = 1
            for post in row_posts_list:
                post = post.strip("1.").strip() # remove any prepended "1." (that's the only case the regex split doesn't take care of), and then any prepended space/tab characters and any appended newline(s)
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


                    if j in temp_arr: # the jth post in this row is marked as hate speech
                        Y = np.append(Y, 1)
                        row_resps = ast.literal_eval(data[i,3])
                        responses.append(random.choice(row_resps))
                    else: # the jth post in this row is marked as not hate speech
                        Y = np.append(Y, 0)
                else: # it's 'n/a', which gets parsed as nan apparently. So none of these posts are marked as hate
                    Y = np.append(Y, 0)
                j += 1
        print("100%")
        process_responses(responses)
        # print(responses[0])
        # print(responses[1])
        # print(responses[2])
        # print(responses[3])
        counts = vectorizer.fit_transform(list_of_all_posts) # counts in a 2D matrix
        counts_np = np.array(counts.toarray()) # convert to normal numpy format

        feature_names = vectorizer.get_feature_names() # the 1D python list of features (i.e. words) that correspond to the columns of counts_np
        feature_names_np = np.array(feature_names) # convert to numpy

        char_vectorizer.fit(list_of_all_posts)
        self.post_chars = char_vectorizer.get_feature_names() # a 1D python list of all the characters used in the processed posts

        char_vectorizer.fit(responses)
        self.resp_chars = char_vectorizer.get_feature_names() # a 1D python list of all the characters used in the processed responses

        self.list_of_all_posts = list_of_all_posts

        # remove unique features/columns (i.e. words that appear only in one post throughout the corpus)
        non_unique_indeces = np.nonzero(np.count_nonzero(counts_np,axis=0)>1)[0] # the column indeces of the features that appear in more than one document throughout the corpus
        non_unique_counts_np = counts_np[:,non_unique_indeces] # select only the columns at those indeces
        non_unique_feature_names_np = feature_names_np[non_unique_indeces] # select only the feature names at those indeces

        return non_unique_counts_np, non_unique_feature_names_np, Y, responses

        # print(np.array(vectorizer.get_feature_names())[np.nonzero(counts[0])[1]]) # good for seeing the word counts of a single post

    def get_post_chars(self): 
        return self.post_chars
    
    def get_resp_chars(self):
        return self.resp_chars
    
    def get_posts_list(self):
        return self.list_of_all_posts
    
def process_responses(responses):
    for i in range(len(responses)):
        responses[i]= responses[i].strip()
        # responses[i] = responses[i][1:-1] # no longer needed now that we're using ast.literal_eval



def main():
    p = Processor()
    gab_X, gab_feature_names, gab_Y, gab_resps = p.process_files('data/gab.csv', stop_after_rows=5)
    print(p.get_posts_list())
    print(gab_resps)
    print(p.get_post_chars())
    print(p.get_resp_chars())
    # total_counts = gab_X.sum(0)

if __name__ == "__main__":
    main()

