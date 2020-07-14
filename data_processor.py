import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
STOPWORDS = stopwords.words('english')
import re
import random
import ast
import pickle

RANDOM_SEED = 13579

class Processor:

    def process_files(self, *filenames, stop_after_rows=None, overwrite_output_files=True):
        self.max_post_tokens = 0
        self.max_resp_tokens = 0
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
        post_vectorizer = CountVectorizer()
        resp_vectorizer = CountVectorizer(token_pattern=r'(?u)\b\w+\b') # want to keep 1-char words in the responses when tokenizing them
        post_preprocessor = post_vectorizer.build_preprocessor()
        self.post_tokenizer = post_vectorizer.build_tokenizer() # seq2seq also uses this
        self.resp_tokenizer = resp_vectorizer.build_tokenizer() # seq2seq also uses this

        list_of_all_posts = np.empty(0)
        Y = np.empty(0)
        print("Preprocessing progress (by rows of original data):")
        for i in range(posts.shape[0]):
            if i % 100 == 0: print("%.0f%%" % (i*100/posts.shape[0]))
            row_posts_string = post_preprocessor(posts[i]) # preprocess the posts in this row (including making them lowercase)
            row_posts_list = re.split(r'\n\d+\.', row_posts_string) # split up all the posts in a given row
            j = 1
            for post in row_posts_list:
                post = post.strip("1.").strip() # remove any prepended "1." (that's the only case the regex split doesn't take care of), and then any prepended space/tab characters and any appended newline(s)
                post = re.sub(r'\.|,|;|:|\?|!|\(|\)|"|\u201C|\u201D', '', post) # remove certain punctuation

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
                if len(self.post_tokenizer(post)) > self.max_post_tokens: self.max_post_tokens = len(self.post_tokenizer(post))
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
                        row_max_resp_tokens = max(map(lambda resp: len(self.resp_tokenizer(resp)), row_resps))
                        if row_max_resp_tokens > self.max_resp_tokens: self.max_resp_tokens = row_max_resp_tokens
                        responses.append(random.choice(row_resps).lower())
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
        counts = post_vectorizer.fit_transform(list_of_all_posts) # counts in a 2D matrix
        counts_np = np.array(counts.toarray()) # convert to normal numpy format

        feature_names = post_vectorizer.get_feature_names() # the 1D python list of features (i.e. words) that correspond to the columns of counts_np
        feature_names_np = np.array(feature_names) # convert to numpy

        resp_vectorizer.fit(responses)
        resp_tokens = resp_vectorizer.get_feature_names() # a 1D python list of all the tokens (probably words) used in the processed responses
        resp_tokens_np = np.array(resp_tokens)
        
        responses=np.array(responses)

        # remove unique features/columns (i.e. words that appear only in one post throughout the corpus)
        non_unique_indices = np.nonzero(np.count_nonzero(counts_np,axis=0)>1)[0] # the column indices of the features that appear in more than one document throughout the corpus
        non_unique_counts_np = counts_np[:,non_unique_indices] # select only the columns at those indices
        non_unique_feature_names_np = feature_names_np[non_unique_indices] # select only the feature names at those indices

        if overwrite_output_files:
            np.savez_compressed('data/preprocessed_data.npz', post_word_counts=non_unique_counts_np, post_feature_names=non_unique_feature_names_np, post_labels=Y, post_texts=list_of_all_posts, post_tokens=feature_names_np, response_texts=responses, resp_tokens=resp_tokens_np)
            with open('data/preprocessor.pkl', 'wb') as obj_file:
                pickle.dump(self, obj_file, pickle.HIGHEST_PROTOCOL)
        
        return {'post_word_counts': non_unique_counts_np, 'post_feature_names': non_unique_feature_names_np, 'post_labels': Y, 'post_texts': list_of_all_posts, 'post_tokens': feature_names_np, 'response_texts': responses, 'resp_tokens': resp_tokens_np}

        # print(np.array(vectorizer.get_feature_names())[np.nonzero(counts[0])[1]]) # good for seeing the word counts of a single post
    
    def get_max_post_tokens(self):
        return self.max_post_tokens

    def get_max_resp_tokens(self):
        return self.max_resp_tokens
    
    def get_post_tokenizer(self):
        return self.post_tokenizer
    
    def get_resp_tokenizer(self):
        return self.resp_tokenizer
    
def process_responses(responses):
    for i in range(len(responses)):
        responses[i]= responses[i].strip()
        # responses[i] = responses[i][1:-1] # no longer needed now that we're using ast.literal_eval

def load_preprocessor():
    with open('data/preprocessor.pkl', 'rb') as obj_file:
        preprocessor = pickle.load(obj_file)
    return preprocessor

def load_preprocessed_data():
    return np.load('data/preprocessed_data.npz')

def main():
    p = Processor()
    gab_X, gab_feature_names, gab_Y, gab_post_texts, gab_post_tokens, gab_resp_texts, gab_resp_tokens = p.process_files('data/gab.csv', stop_after_rows=500, overwrite_output_files=False).values()
    print(gab_X.shape, gab_feature_names.shape, gab_Y.shape, gab_post_texts.shape, gab_post_tokens.shape, gab_resp_texts.shape, gab_resp_tokens.shape)
    # total_counts = gab_X.sum(0)

if __name__ == "__main__":
    main()

