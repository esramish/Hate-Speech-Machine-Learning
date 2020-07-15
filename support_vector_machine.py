from sklearn.linear_model import SGDClassifier

class SVM:
    
    def fit( self, trainData, trainLabels ):
        self.svm = SGDClassifier()
        self.svm.fit( trainData, trainLabels )
        
    def test( self, testData, testLabels ):
        print( "Accuracy:", self.svm.score( testData, testLabels, sample_weight=None ) )
