from utilize.transform import select_features_by_name
from utilize.test import evaluate_model

class feature_selector():

	def __init__(self, feature_names):

		self.feature_names = feature_names
		self.selected_features = []

	def forward_sequential_selection(self, X_train, y_train, X_test, y_test, model, 
										evaluation = 'BA', report = 'True'):
		'''
		Take splitted dataset and a model, use forward sequential selection to select a 
		subset of features that gives the highest score on given test set. 

		Keyword Arguments:
			X_train, y_train, X_test, y_test: [narray] -- splitted dataset
			model: [sklearn model] -- model used to fit and predict the dataset
			evaluation: [str] -- specify the type of evaluation (score)
			report: [Boolen] -- whether to report the progress

		'''

		self.n_labels = y_train.shape[-1]
		self.evaluation = evaluation

		selected_features = self.selected_features
		remaining_features = self.feature_names.copy()
		for feature in selected_features:
			remaining_features.remove(feature)
		score = 0

		for i in range(len(remaining_features)):
			flag = 0

			# Try all the features that has not been selected
			for j, feature in enumerate(remaining_features):

	        	# Select features
				temp_features = selected_features + [feature]
				feature_selector = select_features_by_name(temp_features, self.feature_names)
				temp_X_train = feature_selector.fit_transform(X_train)
				temp_X_test = feature_selector.transform(X_test)

				# Fit the model
				model.fit(temp_X_train, y_train)

				# predict and calcuate score
				if evaluation == 'model_default':
					temp_score = model.score(temp_X_test, y_test)
				elif evaluation == 'BA':
					_, _, _, temp_score = evaluate_model(model, temp_X_test, y_test, report = False)
				else:
					raise NameError('Evaluation does not exist!')
	            
				# If the score increases, update 
				if temp_score > score:

					score = temp_score
					added_feature = feature
					flag = 1
				print('Try feature: %s\t[%d/%d]\t score: %f' %(feature, j, len(remaining_features), temp_score))
	                
			if flag == 0:
				break

	        # Update current selected features and remaining feature to be selected
			selected_features = selected_features + [added_feature]
			remaining_features.remove(added_feature)

			if report:
				print('\nAdd %s\t%d features selected\tscore %f' %(added_feature, len(selected_features), score))
	    
		self.selected_features = selected_features

		return selected_features, score

if __name__ == 'main': 

	None