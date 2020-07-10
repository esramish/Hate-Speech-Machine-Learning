from sklearn.svm import LinearSVC

class SVM:
    
    def fit( self, trainData, trainLabels ):
        self.svm = LinearSVC( random_state=0, tol=1e-5 )
        self.svm.fit( trainData, trainLabels )
        
    def test( self, testData, testLabels ):
        predLabels = self.svm.predict( testData )
        print( "Accuracy:", self.svm.score( testData, testLabels, sample_weight=None ) )
