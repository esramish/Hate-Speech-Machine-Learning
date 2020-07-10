from data_processor import *
from support_vector_machine import *
from SVM_performance_tester import *
from sklearn.model_selection import train_test_split

def main():
    processor = Processor()
    data, feature_names, responses, labels = processor.process_files('data/gab.csv', 'data/reddit.csv', stop_after_rows=500)
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, random_state=6)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=6)
    # model = SVM()
    # model.fit(X_train,y_train)
    # model.test(X_val, y_val)
    performance_tester = SVM_performance_tester()
    print('Fitting Models...')
    performance_tester.fit(X_train,y_train)
    print('Done Fitting Models.')
    print('Testing Models...')
    performance_tester.test(X_val,y_val)


if __name__ == "__main__":
    main()