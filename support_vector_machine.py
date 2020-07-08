from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from data_processor import *


processor = Data_processor()
data, labels = processor( file )


def makeSVM( trainData, trainLabels ):
    svm = LinearSVC( random_state=0, tol=1e-5 )
    svm.fit( trainData, trainLabels )
    
    return svm
    
    
def testSVM( testData ):
    predLabels = svm.predict( testData )
    print( "Accuracy:", score( testData, testLabels, sample_weight=None ) )
    