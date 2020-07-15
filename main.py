from data_processor import *
from support_vector_machine import *
from SVM_performance_tester import *
from sklearn.model_selection import train_test_split

def main():
    # only used when we don't have already-processed data stored in files:
    # processor = Processor()
    # data, feature_names, labels, post_texts, post_tokens, response_texts, resp_tokens = processor.process_files('data/gab.csv', 'data/reddit.csv').values() 

    # only used when we do have already-processed data stored in files:
    processor = load_preprocessor('gab_500_')
    data, feature_names, labels, post_texts, post_tokens, response_texts, resp_tokens = load_preprocessed_data('gab_500_').values() 

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, random_state=6)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=6)

    #Model uses a SDG as it has the best accuracy
    model = SVM()
    model.fit(X_train,y_train)
    model.test(X_val, y_val)

    # RUN performance tester to test the accuracy of different models
    
    # performance_tester = SVM_performance_tester(includeSVCs=True)
    # print('\n\nFitting Models...')
    # performance_tester.fit(X_train,y_train)
    # print('Done Fitting Models.\n\n')
    # print('Testing Models on validation set...')
    # performance_tester.test(X_val,y_val)
    # print('\n\nTesting Models on test set...')
    # performance_tester.test(X_test,y_test)


if __name__ == "__main__":
    main()
