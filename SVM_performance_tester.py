from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from datetime import datetime # for timing parts of program


class SVM_performance_tester:
    
    def __init__( self, includeSVCs=True ):
        self.includeSVCs = includeSVCs

    def fit( self, trainData, trainLabels ):
        #Test different variations fo Linear SVC
        self.linear_svm1 = LinearSVC( random_state=0, tol=1e-5, max_iter=5000 )
        self.linear_svm2 = LinearSVC( random_state=0, max_iter=5000 )
        self.linear_svm3 = LinearSVC( tol=1e-5, max_iter=5000 )
        self.linear_svm4 = LinearSVC( loss='hinge', random_state=0, tol=1e-5, max_iter=5000 )
        self.linear_svm5 = LinearSVC( loss='hinge', random_state=0, max_iter=5000 )
        self.linear_svm6 = LinearSVC( loss='hinge', tol=1e-5, max_iter=5000 )
        self.linear_svm7 = LinearSVC( loss='hinge', max_iter=5000 )

        if self.includeSVCs:
            self.svc1 = SVC( kernel='linear' )
            # self.svc2 = SVC( kernel='poly' )
            self.svc3 = SVC( kernel='rbf' )
            self.svc4 = SVC( kernel='sigmoid' )
            # self.svc5 = SVC( kernel='poly', gamma='auto' )
            # self.svc6 = SVC( kernel='rbf', gamma='auto' )
            # self.svc7 = SVC( kernel='sigmoid', gamma='auto' )

        self.sgd1 = SGDClassifier( loss='squared_hinge' )
        self.sgd2 = SGDClassifier( loss='squared_hinge', learning_rate='adaptive', eta0=0.001 )
        self.sgd3 = SGDClassifier( loss='perceptron' )
        self.sgd4 = SGDClassifier( loss='perceptron', learning_rate='adaptive', eta0=0.001 )
        self.sgd5 = SGDClassifier( learning_rate='adaptive', eta0=0.001 )
        self.sgd6 = SGDClassifier()

        print('Fitting Linear SVMs...')
        self.linear_svm1.fit( trainData, trainLabels )
        self.linear_svm2.fit( trainData, trainLabels )
        self.linear_svm3.fit( trainData, trainLabels )
        self.linear_svm4.fit( trainData, trainLabels )
        self.linear_svm5.fit( trainData, trainLabels )
        self.linear_svm6.fit( trainData, trainLabels )
        self.linear_svm7.fit( trainData, trainLabels )
        print('Done.\n')
        if self.includeSVCs:
            print('Fitting SVCs...')
            self.svc1.fit( trainData, trainLabels )
            print('Done fitting SVC1 at',datetime.now())
            # self.svc2.fit( trainData, trainLabels )
            self.svc3.fit( trainData, trainLabels )
            print('Done fitting SVC3 at',datetime.now())
            self.svc4.fit( trainData, trainLabels )
            print('Done fitting SVC4 at',datetime.now())
            # self.svc5.fit( trainData, trainLabels )
            # self.svc6.fit( trainData, trainLabels )
            # self.svc7.fit( trainData, trainLabels )
            print('Done.\n')
        print('Fitting SGDs...')
        self.sgd1.fit( trainData, trainLabels )
        self.sgd2.fit( trainData, trainLabels )
        self.sgd3.fit( trainData, trainLabels )
        self.sgd4.fit( trainData, trainLabels )
        self.sgd5.fit( trainData, trainLabels )
        self.sgd6.fit( trainData, trainLabels )
        
    def test( self, testData, testLabels ):

        print( "AccuracyLINEARSVC1:", self.linear_svm1.score( testData, testLabels, sample_weight=None ) )
        print( "AccuracyLINEARSVC2:", self.linear_svm2.score( testData, testLabels, sample_weight=None ) )
        print( "AccuracyLINEARSVC3:", self.linear_svm3.score( testData, testLabels, sample_weight=None ) )
        print( "AccuracyLINEARSVC4:", self.linear_svm4.score( testData, testLabels, sample_weight=None ) )
        print( "AccuracyLINEARSVC5:", self.linear_svm5.score( testData, testLabels, sample_weight=None ) )
        print( "AccuracyLINEARSVC6:", self.linear_svm6.score( testData, testLabels, sample_weight=None ) )
        print( "AccuracyLINEARSVC7:", self.linear_svm7.score( testData, testLabels, sample_weight=None ) )
        print("")

        if self.includeSVCs:
            print( "AccuracySVC1:", self.svc1.score( testData, testLabels, sample_weight=None ) )
            # print( "AccuracySVC2:", self.svc2.score( testData, testLabels, sample_weight=None ) )
            print( "AccuracySVC3:", self.svc3.score( testData, testLabels, sample_weight=None ) )
            print( "AccuracySVC4:", self.svc4.score( testData, testLabels, sample_weight=None ) )
            # print( "AccuracySVC5:", self.svc5.score( testData, testLabels, sample_weight=None ) )
            # print( "AccuracySVC6:", self.svc6.score( testData, testLabels, sample_weight=None ) )
            # print( "AccuracySVC7:", self.svc7.score( testData, testLabels, sample_weight=None ) )
            print("")

        print( "AccuracySGD1:", self.sgd1.score( testData, testLabels, sample_weight=None ) )
        print( "AccuracySGD2:", self.sgd2.score( testData, testLabels, sample_weight=None ) )
        print( "AccuracySGD3:", self.sgd3.score( testData, testLabels, sample_weight=None ) )
        print( "AccuracySGD4:", self.sgd4.score( testData, testLabels, sample_weight=None ) )
        print( "AccuracySGD5:", self.sgd5.score( testData, testLabels, sample_weight=None ) )
        print( "AccuracySGD6:", self.sgd6.score( testData, testLabels, sample_weight=None ) )

