from textgenrnn import textgenrnn
from data_processor import *
import numpy as np






def main():
    p = Processor()
    gab_X, gab_feature_names, labels, post_texts, post_tokens, responses, resp_tokens = p.process_files('data/gab.csv', stop_after_rows=50, overwrite_output_files=False).values()
    # total_counts = gab_X.sum(0)
    textgen = textgenrnn()
    textgen.train_on_texts(responses,num_epochs = 10)
    textgen.generate(3)

if __name__ == "__main__":
    main()