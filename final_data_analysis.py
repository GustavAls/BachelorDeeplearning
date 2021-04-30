

import numpy as np
import pandas as pd
import sklearn.preprocessing
import utils
from sklearn.utils import class_weight
import psutil
import matplotlib.pyplot as plt
import os
import pickle
import sklearn.metrics as metrics
from scipy import io
import seaborn as sns
# import make_plots
from sklearn.metrics import auc
from sklearn.metrics import plot_roc_curve
from sklearn.base import BaseEstimator,TransformerMixin
show_training_results = False
show_AISC_results = False
path = r'C:\Users\ptrkm\Bachelor\AISC_results'
average_voting = True
majority_voting = False



class IdentityTransformer(BaseEstimator,TransformerMixin):
    def __init__(self):
        self._estimator_type = "classifier"
    def fit(self,input_array, y = None):
        return self
    def transform(self, input_array,y = None):
        return input_array*1
    def decision_function(self,input_array, y = None):
        return input_array*1



class data_visualiser:
    def __init__(self, save_path = None):
        if save_path is not None:
            self.save_path = save_path
            self.results = {}

        self.ground_path = r'C:\Users\ptrkm\Bachelor'
        self.path_to_train_ss = r'C:\Users\ptrkm\OneDrive\Dokumenter\Bachelor deep learning'
        self.path_to_train_rr = r'C:\Users\ptrkm\OneDrive\Dokumenter\Bachelor deep learning\rr_results'

        self.labels = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']
        self.isic_labels = pd.read_csv(os.path.join(self.ground_path,'ISIC_2019_Training_GroundTruth.csv'))
        self.aisc_labels = pd.read_csv(os.path.join(self.ground_path,'labels.csv'))
        self.aisc_isic_labels = pd.read_csv(os.path.join(self.ground_path,'labels_aisc_isic.csv'))

        self.isic_indices = pickle.load(open(os.path.join(self.ground_path,'indices_isic2019_one_cv.pkl'),'rb'))
        self.aisc_indices = pickle.load(open(os.path.join(self.ground_path,'indices_AISC.pkl'),'rb'))
        self.aisc_isic_indices = pickle.load(open(os.path.join(self.ground_path,'indices_aisc_plus_isic.pkl'),'rb'))

        self.indices_dict = {'aisc': self.aisc_indices,
                             'isic': self.isic_indices,
                             'combined': self.aisc_isic_indices}
        self.pkl_list = []
        self.weighted_accuracy = None
        self.accuracy = None
        self.auc = None
        self.confusion_matrix = None
        self.rr_list = []
        self.ss_list = []

    def create_pkl_list(self, network_list):
        """
        :param network_list: list of networks
        :return: list containing pkl files of networks
        """
        self.rr_list = [i for i in network_list if 'rr' in i]
        self.ss_list = [i for i in network_list if 'ss' in i]

        if len(self.ss_list)+len(self.rr_list) == 1:
            self.rr_list = self.rr_list+self.ss_list
            if 'rr' in self.rr_list:
                files_list = os.listdir(self.path_to_train_rr)
                chosen_path = self.path_to_train_rr
            else:
                files_list = os.listdir(self.path_to_train_ss)
                chosen_path = self.path_to_train_ss
            for files in files_list:
                if self.rr_list[0] in files:
                    chosen_path = os.path.join(chosen_path,files)
                    for fls in os.listdir(chosen_path):
                        if '.pkl' in fls:
                            pcl = pickle.load(open(os.path.join(chosen_path, fls),'rb'))
                            if len(pcl['bestPred']) > 6:
                                self.pkl_list = pcl
                            else:
                                self.pkl_list = {'bestPred':pcl['bestPred'][0], 'targets':pcl['targets'][0]}

        else:
            for rrs in self.rr_list:
                for files in os.listdir(self.path_to_train_rr):
                    if rrs in files:
                        chosen_path = os.path.join(self.path_to_train_rr, files)
                        for fls in os.listdir(chosen_path):
                            if '.pkl' in fls:
                                pcl = pickle.load(open(os.path.join(chosen_path, fls),'rb'))
                                if len(pcl['bestPred']) > 6:
                                    self.pkl_list.append(pcl)
                                else:
                                    new_dict = {'bestPred':pcl['bestPred'][0], 'targets':pcl['targets'][0]}
                                    self.pkl_list.append(new_dict)
            for ssr in self.ss_list:
                for files in os.listdir(self.path_to_train_ss):
                    if ssr in files:
                        chosen_path = os.path.join(self.path_to_train_ss, files)
                        for fls in os.listdir(chosen_path):
                            if '.pkl' in fls:
                                pcl = pickle.load(open(os.path.join(chosen_path, fls), 'rb'))
                                if len(pcl['bestPred']) > 6:
                                    self.pkl_list.append(pcl)
                                else:
                                    new_dict = {'bestPred':pcl['bestPred'][0], 'targets':pcl['targets'][0]}
                                    self.pkl_list.append(new_dict)



    def create_pkl(self, network_names = None, create_ensemble = True, evaluation_set=None):
        """
        :param pkl_list:
        :param create_ensemble:
        :param evaluation_set:
        :return:
        """
        #Creates a combined pickle file for further metrics.
        #pkl_list is either list of pickle files or one

        if type(self.pkl_list) is list:
            if create_ensemble:
                for idx, pkl in enumerate(self.pkl_list):
                    if 'targets' in pkl.keys() and idx == 0:
                        targets = pkl['targets']
                        predictions = np.zeros(targets.shape)
                        predictions += pkl['bestPred']/len(self.pkl_list)
                    elif 'targets' in pkl.keys():
                        predictions+= pkl['bestPred']/len(self.pkl_list)
                    elif 'targets' not in pkl.keys() and 'targets' in locals():
                        print(" number {} in the list should not be included".format(idx))
                    elif 'targets' not in pkl.keys() and idx == 0:
                        predictions = pkl['extPred']
                    elif 'targets' not in pkl.keys():
                        predictions += pkl['extPred']
            else:
                targets = []
                predictions = []
                for idx, pkl in enumerate(self.pkl_list):
                    if 'targets' in pkl.keys():
                        targets.append(pkl['targets'])
                        predictions.append(pkl['bestPred'])
                    elif 'targets' not in pkl.keys():
                        predictions.append(pkl['extPred'])
        else:
            if 'targets' in self.pkl_list.keys():
                predictions = self.pkl_list['bestPred']
                targets = self.pkl_list['targets']

        if 'targets' not in locals():
            if evaluation_set == 'aisc':
                targets = self.aisc_labels[self.indices_dict['aisc']['valIndCV']]
            elif evaluation_set == 'isic':
                targets = self.isic_labels[self.indices_dict['isic']['valIndCV']]
            elif evaluation_set == 'combined':
                targets = self.aisc_isic_labels[self.indices_dict['combined']['valIndCV']]


        self.pkl = {'predictions': predictions,
                    'targets': targets,
                    'names': network_names}


    def score_metrics(self):
        # calculates confusion matrix, weighted accuracy, accuracy and AUC for given pkl

        if type(self.pkl['predictions']) is not list:
            self.weighted_accuracy = metrics.balanced_accuracy_score(np.argmax(self.pkl['targets'],1),
                                                                     np.argmax(self.pkl['predictions'],1))

            self.accuracy = metrics.accuracy_score(np.argmax(self.pkl['targets'],1),
                                                   np.argmax(self.pkl['predictions'],1))

            self.confusion_matrix = metrics.confusion_matrix(np.argmax(self.pkl['targets'],1),
                                                                   np.argmax(self.pkl['predictions'],1))
            self.results = {'ensemble_results': {
                'weighted_accuracy': self.weighted_accuracy,
                'accuracy': self.accuracy,
                'confusion_matrix': self.confusion_matrix
            }}
            self.make_confusion_matrix(cf = self.confusion_matrix,categories=self.labels,cbar = False, figsize=(10,8))


        else:
            self.weighted_accuracy = [metrics.balanced_accuracy_score(np.argmax(preds,1), np.argmax(tags,1))
                                      for preds, tags in zip(self.pkl['predictions'],self.pkl['targets'])]

            self.weighted_accuracy = [metrics.accuracy_score(np.argmax(preds, 1), np.argmax(tags, 1))
                                      for preds, tags in zip(self.pkl['predictions'], self.pkl['targets'])]

            self.weighted_accuracy = [metrics.confusion_matrix(np.argmax(preds, 1), np.argmax(tags, 1))
                                      for preds, tags in zip(self.pkl['predictions'], self.pkl['targets'])]
            self.make_roc_curve()



    def make_roc_curve(self):
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        classifier = IdentityTransformer()
        fig, ax = plt.subplots()
        for i, (predictions, targets) in enumerate(zip(self.pkl['predictions'],self.pkl['targets'])):
            classifier.fit(np.argmax(predictions,1),np.argmax(targets,1))
            viz = plot_roc_curve(classifier, np.argmax(predictions,1), np.argmax(targets,1),
                                 name=self.pkl['names'][i],
                                 alpha=0.3, lw=1, ax=ax)
            interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(viz.roc_auc)

        ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                label='Chance', alpha=.8)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        ax.plot(mean_fpr, mean_tpr, color='b',
                label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                lw=2, alpha=.8)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                        label=r'$\pm$ 1 std. dev.')

        ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
               title="Receiver operating characteristic example")
        ax.legend(loc="lower right")
        plt.show()


    def make_confusion_matrix(self, cf,
                              group_names=None,
                              categories='auto',
                              count=True,
                              percent=True,
                              cbar=True,
                              xyticks=True,
                              xyplotlabels=True,
                              sum_stats=True,
                              figsize=None,
                              cmap='Blues',
                              title=None,
                              ):
        '''
        This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
        Arguments
        ---------
        cf:            confusion matrix to be passed in
        group_names:   List of strings that represent the labels row by row to be shown in each square.
        categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
        count:         If True, show the raw number in the confusion matrix. Default is True.
        normalize:     If True, show the proportions for each category. Default is True.
        cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                       Default is True.
        xyticks:       If True, show x and y ticks. Default is True.
        xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
        sum_stats:     If True, display summary statistics below the figure. Default is True.
        figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
        cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                       See http://matplotlib.org/examples/color/colormaps_reference.html

        title:         Title for the heatmap. Default is None.
        '''

        # CODE TO GENERATE TEXT INSIDE EACH SQUARE
        blanks = ['' for i in range(cf.size)]

        if group_names and len(group_names) == cf.size:
            group_labels = ["{}\n".format(value) for value in group_names]
        else:
            group_labels = blanks

        if count:
            group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
        else:
            group_counts = blanks

        if percent:
            weighted = cf.diagonal()/np.sum(cf,axis =1)
            group_percentages = []
            count = 0
            for i in cf.flatten():
                if i not in cf.diagonal():
                    group_percentages.append('')
                else:
                    group_percentages.append("{0:.2%}".format(weighted[count]))
                    count += 1

            # group_percentages = ["{0:.2%}".format(weighted[idx // cf.shape[0]]) if idx%cf.shape[1]==0 else '' for idx, value in enumerate(cf.flatten())]
        else:
            group_percentages = blanks

        box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels, group_counts, group_percentages)]
        box_labels = np.asarray(box_labels).reshape(cf.shape[0], cf.shape[1])

        # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
        if sum_stats:
            # Accuracy is sum of diagonal divided by total observations
            accuracy = np.trace(cf) / float(np.sum(cf))

            # if it is a binary confusion matrix, show some more stats
            if len(cf) == 2:
                # Metrics for Binary Confusion Matrices
                precision = cf[1, 1] / sum(cf[:, 1])
                recall = cf[1, 1] / sum(cf[1, :])
                f1_score = 2 * precision * recall / (precision + recall)
                stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                    accuracy, precision, recall, f1_score)
            else:
                stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
        else:
            stats_text = ""

        # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
        if figsize == None:
            # Get default figure size if not set
            figsize = plt.rcParams.get('figure.figsize')

        if xyticks == False:
            # Do not show categories if xyticks is False
            categories = False

        # MAKE THE HEATMAP VISUALIZATION
        plt.figure(figsize=figsize)
        for i in range(cf.shape[0]):
            cf[i,:] = cf[i,:]/np.sum(cf[i,:])*20

        sns.heatmap(cf, annot=box_labels, fmt="", cmap=cmap, cbar=cbar, xticklabels=categories, yticklabels=categories)

        if xyplotlabels:
            plt.ylabel('True label')
            plt.xlabel('Predicted label' + stats_text)
        else:
            plt.xlabel(stats_text)

        if title:
            plt.title(title)
        plt.show()


def main():
    our_ensemble = ['res_101_rr', 'efficientnet_b5_rr', 'se_resnet101_rr', 'nasnetamobile_rr',
                    'efficientnet_b6_ss', 'resnext_101_32_8_wsl_rr', 'dense_169_rr']

    their_ensemble = ['efficientnet_b0_rr', 'efficientnet_b1_rr', 'efficientnet_b2_rr', 'efficientnet_b3_rr',
                      'efficientnet_b4_rr', 'efficientnet_b5_rr', 'efficientnet_b0_ss', 'efficientnet_b1_ss',
                      'efficientnet_b2_ss', 'efficientnet_b3_ss', 'efficientnet_b4_ss', 'efficientnet_b5_ss',
                      'efficientnet_b6_ss', 'senet154_ss']

    gustav_bunder = data_visualiser()
    gustav_bunder.create_pkl_list(our_ensemble)
    gustav_bunder.create_pkl(network_names=our_ensemble, create_ensemble=False)
    gustav_bunder.score_metrics()
main()