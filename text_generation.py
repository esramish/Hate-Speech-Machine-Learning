from textgenrnn import textgenrnn
from data_processor import *
import numpy as np






def main():
    processor = load_preprocessor('gab_500_')
    gab_X, gab_feature_names, labels, post_texts, post_tokens, responses, resp_tokens = load_preprocessed_data('gab_500_').values()
    textgen = textgenrnn()
    textgen.train_on_texts(responses,num_epochs = 10)
    textgen.generate(3)

if __name__ == "__main__":
    main()