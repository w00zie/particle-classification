from __future__ import print_function
from __future__ import division

from sklearn.model_selection import train_test_split, cross_val_score, learning_curve, GridSearchCV, StratifiedKFold, ShuffleSplit
#from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *
from sklearn.tree import DecisionTreeClassifier
import sklearn.tree as tree
from time import clock
from IPython.display import Image
import pydotplus
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools

class gamma(object):

    def __init__(self, target_name, gamma, hadron):
    	
    	self._target_name = target_name
    	self._classifier = dict()
    	self._dataset = None
    	self._train = (None, None)
    	self._test = (None, None)
    	self._gamma = gamma
    	self._hadron = hadron
    	
    
    @property
    def columns(self):
    	"""
    	Returns data set split in features and targets (X and y)
    	"""
        return (self.features, self.targets)
    
    @property
    def feature_names(self):
        """
        Dataset columns names excluding targets
        """
    	columns = self._dataset.columns.tolist()
    	return [c for c in columns if c is not self._target_name]

    @property
    def target_names(self):
        """
        Dataset columns names excluding features
        """
    	columns = self._dataset.columns.tolist()
        return [c for c in columns if c is self._target_name]

    @property
    def features(self):
        """
        Dataset features columns
        """
        
        return self._dataset.select(lambda x: x != self._target_name, axis = 1)
    
    @property
    def targets(self):
        """
        Dataset targets columns
        """
        return self._dataset[self._target_name].values
    
    
    def binarize_class(self):
        """
        Substitute 0,1 to h,g in dataset
        """
    	self._dataset.replace(to_replace=self._gamma, value = 1, inplace = True)
        self._dataset.replace(to_replace=self._hadron, value = 0, inplace = True)
    
    	
    def prepare_data(self, dataframe):
        """
        Load data from dataset and prepare for classification
        """
        print('='*50)
        print('Loading MAGIC Gamma Telescope Dataframe')
        self._dataset = pd.read_csv(dataframe)
        self.binarize_class()
        
    
    def split_dataset(self, percent_testing = 0.2):
        """
        Split dataset for training and testing sets
        """

        print('Executing a random Train/Test split [test size = %.2f]' % (percent_testing) )

        (X, y) = self.columns
        
        X_train, X_test, y_train, y_test = train_test_split(
                                                X,
                                                y,
                                                test_size = percent_testing,
                                                random_state = 42,
                                                )
	
	self._train = (X_train, y_train)
	self._test = (X_test, y_test)
    

    def train(self):
        """
        Execute training with random forest classifier
        """
      	params = dict()

        # init the classifier
      	random_forest = RandomForestClassifier(
      						random_state = 42,
      						n_jobs = -1,
      						n_estimators=13,
      						criterion='entropy',
      						max_depth=13,
      						min_impurity_split=1e-3,
      						#max_features=4,
      						#min_samples_leaf=2,
      						#min_samples_split=4,
      						)
      						
    	decision_tree = DecisionTreeClassifier(
    					#max_depth = 3,
    					#min_samples_leaf=5,
    					random_state = 42
    					)

        # max depth for a tree
#	params['max_depth'] = range(1, 17)	# 2^17 =~ 13k

        # criteria for classification
#	params['criterion'] = ['gini', 'entropy']

        # number of trees in the forest
#	params['n_estimators'] = range(10,20)

	
      	(X_train,y_train) = self._train

        # Grid Search for best parameters with scoring
        
#	scoring_function = make_scorer(average_precision_score)
        scoring_function = make_scorer(roc_auc_score)

#      	grid = GridSearchCV(
#                random_forest,
#                params,
#                cv=10,
#                scoring = scoring_function,
#                )

      	random_forest.fit(X_train,y_train)
#      	grid.fit(X_train,y_train)
#	decision_tree.fit(X_train,y_train)
	
        # set valus from routine to masked attribute
      	self._classifier = random_forest
#      	self._classifier = grid.best_estimator_
#	self._classifier = decision_tree

	print(self._classifier.feature_importances_)
#      	print("\n    Best parameters:")

#      	print("\t", grid.best_params_)
      	

    def test(self):
        """
        Evaluates the classifier score on test data
        """
      	(X_test, y_test) = self._test
      	score_on_test_data = self._classifier.score(X_test, y_test)
      	
    	print('Classifier score : %.3f' % (score_on_test_data) )
        return score_on_test_data
        
         
    def cross_validation(self):
        """
        Execute cross validation on train set with trained classifier
        """

    	print('Executing a 10 Fold Cross Validation')

    	classifier = self._classifier
    	(X_train, y_train) = self._train

    	score_array = cross_val_score(
    			        classifier,
                                X_train,
                                y_train, 
    				scoring = make_scorer(roc_auc_score), 
    				cv = 10,
                                )

    	print('Scores array : ', score_array )
        
    ######################################################################################
    
    def plot_tree(self):
    	"""
    	Generates a .dot file containing the tree fitted on data. 
    	Convert it to png using 'dot -Tpng tree.dot -o tree.png'
    	Info: http://scikit-learn.org/stable/modules/tree.html
    	"""
    	
    	feature_names = ['fLength','fWidth','fSize','fConc','fConc1','fAsym',
    			'fM3Long','fM3Trans','fAlpha','fDist']
    	class_names = ['hadron','gamma']		
	dot_data = tree.export_graphviz(self._classifier, out_file='tree.dot',
			feature_names=feature_names,
			class_names=class_names,
                        filled=True, rounded=True,  
                        special_characters=True)
                           
    
    def plot_roc(self):
        """
        Plot the Receiver Operating Characteristic with Matplotlib
        Info : http://scikit-learn.org/stable/modules/model_evaluation.html#roc-metrics 
        """

    	print('Plotting the ROC Curve...\n')

        (X_test, y_test) = self._test
	
        prob_prediction_for_y = self._classifier.predict_proba(X_test)[:,1]
	
        false_positive_rate, true_positive_rate, _ = roc_curve(
                y_test,
                prob_prediction_for_y
                )

        area_under_roc = roc_auc_score(y_test, prob_prediction_for_y)
        
	area_under_pr = average_precision_score(y_test, prob_prediction_for_y) 	
	
        print('Area under ROC : %0.3f' % area_under_roc)
        print('Area under Precision-Recall : %0.3f' % area_under_pr)

        plt.plot(
                false_positive_rate,
                true_positive_rate,
                label = 'area = %0.3f' % auc(false_positive_rate, true_positive_rate),
                )
          
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
	plt.plot([0, 1], [0, 1], 'k--')
	
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')

        plt.title('Receiver Operating Characteristic (ROC)')

        plt.legend(loc="lower right")

	plt.savefig('roc_curve.png', bbox_inches='tight')
#	plt.show()
    
    
    
    def plot_learning_curve(self, estimator, title, ylim = None, cross_validator = None,
                    n_jobs = -1, train_sizes = np.linspace(.1, 1.0, 5)):
                    
        """
        Generate a simple plot of the learning curve.
        Source : http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
        """
        
        print('Plotting the Learning Curve...\n')
        
	X_train, y_train = self._train

        plt.figure()
        plt.title(title)

        if ylim is not None:
            plt.ylim(*ylim)

        plt.xlabel("Training examples")
        plt.ylabel("Error")

        train_sizes, train_scores, test_scores = learning_curve(
                                                            estimator,
                                                            X_train,
                                                            y_train,
                                                            cv = 10,
                                                            n_jobs = n_jobs,
                                                            train_sizes = np.linspace(0.1,1.,10),
                                                        )
                                                        
        train_scores_mean = np.mean(train_scores, axis=1)
        train_error_mean = 1 - train_scores_mean
        
        train_scores_std = np.std(train_scores, axis=1)
        

        test_scores_mean = np.mean(test_scores, axis=1)
        test_error_mean = 1 - test_scores_mean
        
        test_scores_std = np.std(test_scores, axis=1)
        

        plt.grid()               	
	
	# plot error bars on score

#	plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
#	                  train_scores_mean + train_scores_std, alpha=0.1,
#	                color="r")

#	plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
#	                 test_scores_mean + test_scores_std, alpha=0.1, 
#                    	color="g")

        plt.plot(train_sizes, train_error_mean, 'o-', color="r",
                 label="Training error")

        plt.plot(train_sizes, test_error_mean, 'o-', color="g",
                 label="Cross-validation error")

        plt.legend(loc="best")

	plt.savefig('learn_curve.png')
#	plt.show()   
       	
        
    def plot_confusion_matrix(self, classes, title):
	"""
        This function plots the confusion matrix and prints the Precision and Recall values.
        Source : http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
	"""
	print('Plotting the Confusion Matrix...\n')
	
	(X_test, y_test) = self._test

        y_predicted = self._classifier.predict(X_test)
            
        cm = confusion_matrix(y_test, y_predicted)
        tn, fp, fn, tp = cm.ravel()
        precision = tp/(tp+fp)
        recall = tp/(tp + fn)
        print('Precision = %0.3f' % (precision))
        print('Recall = %0.3f' % (recall))
        
	plt.figure()    
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)

        thresh = cm.max() / 2.
        
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
	    plt.text(j, i, cm[i, j],
	             horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
	
	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')

	np.set_printoptions(precision=2)

	plt.savefig('conf_matrix.png')
#	plt.show()
        
	
    ######################################################################################
	
    
    def __call__(self):
    
    	self.split_dataset(0.2)
        self.train()
#	self.plot_tree()
      	self.plot_roc()
      	
      	self.plot_confusion_matrix(classes=['hadron','gamma'],
        	                  title='Confusion matrix')

      	cross_validator = StratifiedKFold(
                n_splits = 10,
#	        test_size = 0.2,
                random_state = 0,
                )

      	learning_curve_plot = self.plot_learning_curve(
                self._classifier,
                'Learning Curve',
                ylim=[0., 0.3]
#	        cross_validator = cross_validator,
                )
                
	self.test()
	
      	
      	
      	
