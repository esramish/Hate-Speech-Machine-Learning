from data_processor import *
from support_vector_machine import *
from SVM_performance_tester import *
from sklearn.model_selection import train_test_split

def main():
    processor = Processor()
    data, feature_names, labels, responses = processor.process_files('data/gab.csv', 'data/reddit.csv')
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, random_state=6)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=6)
    # model = SVM()
    # model.fit(X_train,y_train)
    # model.test(X_val, y_val)
    performance_tester = SVM_performance_tester(includeSVCs=True)
    print('\n\nFitting Models...')
    performance_tester.fit(X_train,y_train)
    print('Done Fitting Models.\n\n')
    print('Testing Models on validation set...')
    performance_tester.test(X_val,y_val)
    print('\n\nTesting Models on test set...')
    performance_tester.test(X_test,y_test)


if __name__ == "__main__":
    main()
