from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix
from sklearn.metrics import accuracy_score
import numpy as np 

def evaluate_model(model, X_test, y_test, M_test = None, report = True):
	'''
	Estimate a model
	Keyword Arguments:
		model: model to be estimated
		X_test, y_test: test feature matrix and test label matrix. Label matrix should be in multi-class format
		M_test: missing label matrix 
		reprot: whether to print the scores
	Return:
		accuracy: 
		sensitivity: TP / (TP + FN)
		specificity: TN / (TN + FP)
		BA: Averaged mean of sensitivity and specificity
	'''

	y_pred = model.predict(X_test)

	mcm = []
	for i in range(y_test.shape[1]):
		if M_test is not None:
			cm = confusion_matrix(y_test[:,i].T, y_pred[:,i].T, sample_weight = M_test[:,i].T)
		else:
			cm = confusion_matrix(y_test[:,i].T, y_pred[:,i].T)
		cm = np.expand_dims(cm, axis = 0)
		mcm.append(cm)
    
	mcm = np.concatenate(mcm, axis = 0)
	tn = mcm[:, 0, 0]
	tp = mcm[:, 1, 1]
	fn = mcm[:, 1, 0]
	fp = mcm[:, 0, 1]
	
	sensitivity = tp / (tp + fn)
	specificity = tn / (tn + fp)
	BA = (sensitivity + specificity)/2
	accuracy = (tn + tp)/(tn + tp + fn + fp)

	sensitivity = np.sum(sensitivity)/sensitivity.shape[0]
	specificity = np.sum(specificity)/specificity.shape[0]
	BA = np.sum(BA)/BA.shape[0]
	accuracy = np.sum(accuracy)/accuracy.shape[0]

	if report:
		print('%-15s%-15s%-15s%-15s' %('accuaracy', 'sensitivity', 'specificity', 'BA'))
		print('%-15f%-15f%-15f%-15f' %(accuracy, sensitivity, specificity, BA))

	return accuracy, sensitivity, specificity, BA



