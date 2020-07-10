from textgenrnn import textgenrnn
from data_processor import *
import numpy as np






def main():
    p = Processor()
    gab_X, gab_feature_names, responses, gab_Y = p.process_files('data/gab.csv', stop_after_rows=50)
    # total_counts = gab_X.sum(0)
    textgen = textgenrnn()
    textgen.train_on_texts(responses,num_epochs = 10)
    textgen.generate_samples(3)

if __name__ == "__main__":
    main()