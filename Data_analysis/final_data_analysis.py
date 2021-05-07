

import numpy as np
import pandas as pd
import sklearn.preprocessing
from numba.cuda.cudaimpl import ptx_clz

import utils
from sklearn.utils import class_weight
import psutil
import matplotlib.pyplot as plt
import os
import pickle
import sklearn.metrics as metrics
from scipy import io
import seaborn as sns
import make_plots
from sklearn.metrics import auc
from sklearn.metrics import plot_roc_curve
from sklearn.base import BaseEstimator,TransformerMixin
show_training_results = False
show_AISC_results = False
path = r'C:\Users\ptrkm\Bachelor\AISC_results'
average_voting = True
majority_voting = False
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier


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

        self.ensemble_pkls = pickle.load(open(os.path.join(self.ground_path, 'ensemble_results.pkl'),'rb'))

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
        self.confusion_matrix_name = None
        self.ground_frame = None

    def create_pkl_list(self, network_list):
        """
        :param network_list: list of networks
        :return: list containing pkl files of networks
        """
        our_ensemble = ['res_101_rr', 'efficientnet_b5_rr', 'se_resnet101_rr', 'nasnetamobile_rr',
                        'efficientnet_b6_ss', 'resnext_101_32_8_wsl_rr', 'dense_169_rr']

        their_ensemble = ['efficientnet_b1_rr', 'efficientnet_b2_rr', 'efficientnet_b3_rr',
                          'efficientnet_b4_rr', 'efficientnet_b5_rr', 'efficientnet_b0_ss', 'efficientnet_b1_ss',
                          'efficientnet_b2_ss', 'efficientnet_b3_ss', 'efficientnet_b4_ss', 'efficientnet_b5_ss',
                          'efficientnet_b6_ss', 'senet154_ss']
        if type(network_list) is str:
            if network_list in globals():
                network_list = globals()[network_list]
            elif network_list in locals():
                network_list = locals()[network_list]

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
                                print(os.path.join(chosen_path,fls))
                                if len(pcl['bestPred']) > 6:
                                    self.pkl_list.append(pcl)
                                else:
                                    new_dict = {'bestPred':pcl['bestPred'][0], 'targets':pcl['targets'][0]}
                                    self.pkl_list.append(new_dict)
                                print(rrs, self.pkl_list[-1]['bestPred'].shape)
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
                                print(ssr, self.pkl_list[-1]['bestPred'].shape)

    def create_prediction_pkl(self, location,multiple = False, evaluation_set = 'ISIC', network_name = None):

        if not multiple:
            if 'C:' not in location:
                pcl = pickle.load(open(os.path.join(self.ground_path,location),'rb'))
            else:
                pcl = pickle.load(open(location,'rb'))

            predictions = pcl['extPred'][0]
            if evaluation_set == 'ISIC':
                targets = self.isic_labels.drop(['image'],axis = 1).values
            elif evaluation_set == 'AISC':
                targets = self.aisc_labels.drop(['image'],axis =1).values

            name = location
            self.pkl = {'predictions': predictions,
                        'targets': targets,
                        'names': name}
        if multiple:
            if 'C:' not in location:
                pcl = pickle.load(open(os.path.join(self.ground_path,location),'rb'))
            else:
                pcl = pickle.load(open(location,'rb'))

            predictions = pcl['extPred'][0]
            if network_name is not None:
                name = network_name
                self.pkl = {'predictions': predictions,
                            'name': name}


    def create_pkl_new(self, network_names = None, create_ensemble = True, path_to_files = None,prefix = None,
                       format = '.jpg',create_test = False,name = None,num_classes = 8,average_voting=True):
        """

        :param network_names: Names of different networks in the ensemble
        :param create_ensemble: If the function should return ensemble predictions or list of individual
        :param path_to_files: path to evaluation pkl files
        :return:
        """
        path = path_to_files
        count = 0
        image_dict = {}
        for idx, networks in enumerate(network_names):
            prediction_frame = pd.DataFrame()
            image_list =[]
            prediction_list = []
            for pkls in os.listdir(path):
                if networks in pkls:
                    current_pcl = pickle.load(open(os.path.join(path,pkls),'rb'))
                    for counter, img in enumerate(pd.unique(current_pcl['all_images'][0])):
                        if format in img:
                            image = img.removeprefix(prefix)
                            image = image.removesuffix(format)
                            image_list.append(image)
                            prediction_list.append(current_pcl['extPred'][0][counter,:])
                    prediction_frame['image'] = image_list

                    for i in range(np.min([current_pcl['extPred'][0].shape[1],num_classes])):
                        prediction_frame['class'+str(i)] = np.asarray(prediction_list)[:,i]
                    prediction_frame.set_index('image',inplace=True)
                    image_dict[networks] = prediction_frame.transpose()
                    print(networks + 'Was included in the ensemble')


        for idx, networks in enumerate(image_dict.keys()):
            if average_voting:
                if idx == 0:
                    ground_frame = image_dict[networks]
                    print("ground frame was set")

                    image_list = ground_frame.columns.tolist()
                else:
                    for counter, images in enumerate(image_dict[networks].columns):
                        extra_array = np.asarray(image_dict[networks][images])[:num_classes]
                        index = image_list.index(images)
                        ground_frame[ground_frame.columns[index]] = (np.asarray(ground_frame[ground_frame.columns[index]])+\
                                                                    extra_array)/len(network_names)
                print(networks + " was includeded in the average voting ensemble")
            else:
                if idx == 0:
                    ground_frame = image_dict[networks]

                    image_list = ground_frame.columns.tolist()
                    for cols, vals in ground_frame.iteritems():
                        ground_frame[cols] = [1.0 if i==np.argmax(vals) else 0.0 for i in range(len(vals))]
                else:
                    for counter, images in enumerate(image_dict[networks].columns):
                        extra_array = np.asarray(image_dict[networks][images])[:num_classes]
                        extra_array = np.asarray([1.0 if i == np.argmax(extra_array) else 0.0 for i,j in enumerate(extra_array)])
                        index = image_list.index(images)
                        ground_frame[ground_frame.columns[index]] = (np.asarray(ground_frame[ground_frame.columns[index]])+\
                                                                    extra_array)

        self.ground_frame = ground_frame

        if not average_voting:
            for cols, vals in self.ground_frame.iteritems():
                self.ground_frame[cols] = [0.99 if i == np.argmax(vals) else 0.0 for i,j in enumerate(vals)]
        if create_test:
            self.ground_frame = self.ground_frame.transpose()
            self.labels += ['UNK']
            test_labels = []
            for i in range(len(self.ground_frame.columns)):
                test_labels.append(self.labels[i])
            self.ground_frame.columns = test_labels
            if '2018' in name and len(self.ground_frame.columns)>num_classes:
                self.ground_frame.drop([self.ground_frame.columns[-1]],axis=1,inplace=True)
            if '2019' in name:
                self.ground_frame[self.labels[-1]] = [0.0 for i in range(len(self.ground_frame))]

            self.ground_frame.to_csv(os.path.join(self.ground_path,name+'.csv'))



    def evaluate_new(self,network_names = None, path_to_files = None,prefix = None,path_to_labels = None,name=None,
                     format = '.jpg',average_voting = True):
        """

        :param network_names: Ensemble network names
        :param path_to_files: path to evaluation pickles
        :param prefix: path to the image on server
        :param path_to_labels: full path to labels.csv file
        :param average_voting: If average or majority voting should be used, False for majority
        :return: A prediction dictionary, consisting of predictions, labels and name of ensemble

        """
        self.create_pkl_new(network_names=network_names,path_to_files=path_to_files,prefix=prefix,format=format,
                            average_voting=average_voting)
        labels = pd.read_csv(path_to_labels)
        labels.set_index('image', inplace=True)
        labels = labels.transpose()
        final_predictions = pd.DataFrame()
        image_list = self.ground_frame.columns.tolist()
        for columns in labels.columns:
            try:
                final_predictions[columns] = self.ground_frame[columns]
            except:
                breakpoint()
        final_predictions = final_predictions.transpose()
        labels = labels.transpose()

        self.pkl = {
            'predictions': final_predictions.values,
            'targets': labels.values,
            'names': name
        }

    def evaluate_new_white(self, network_names = None, path_to_files = None, prefix = None, format = '.jpg',
                           emulator=None,path_to_labels = None,name=None):

        if emulator == 'knn':
            end_list = ['_F_AS', '_F_CS', '_S_AS', '_S_CS',
                               '_T_AS', '_T_CS', '_C_AS', '_C_CS',
                               '_D_AS', '_D_CS']
            end = '_original'
        elif emulator == 'mohan':
            end_list = [str(i) for i in range(6)]
            end = ""

        self.create_pkl_new(network_names=network_names,path_to_files=path_to_files,prefix=prefix, format=format)

        labels = pd.read_csv(path_to_labels)
        labels.set_index('image', inplace=True)
        labels = labels.transpose()
        final_predictions = pd.DataFrame()
        image_list = self.ground_frame.columns.tolist()
        predictions = []
        n_classes = len(self.ground_frame)
        for columns in labels.columns:
            average_preds = []
            for ends in end_list:
                if columns+ends in image_list:
                    average_preds.append(np.asarray(self.ground_frame[columns+ends]))
            if columns + end in image_list:
                average_preds.append(np.asarray(self.ground_frame[columns+end]))

            if len(average_preds) != 6:
                print('Something went horribly wrong here')

            final_predictions[columns] = np.mean(np.asarray(average_preds),axis=0)

        final_predictions = final_predictions.transpose()

        labels = labels.transpose()
        test = final_predictions.values

        self.pkl = {
            'predictions': final_predictions.values,
            'targets': labels.values,
            'names': name
        }


    def create_pkl(self, network_names = None, create_ensemble = True, evaluation_set=None, already_created = None):
        """
        :param pkl_list:
        :param create_ensemble:
        :param evaluation_set:
        :return:
        """
        #Creates a combined pickle file for further metrics.
        #pkl_list is either list of pickle files or one
        if already_created is None:
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
        else:
            self.create_pkl_list(already_created)
            predictions = self.ensemble_pkls[already_created+'_predictions'].drop(['image'],axis =1).values
            targets = self.ensemble_pkls[list(self.ensemble_pkls.keys())[0]].drop(['image'], axis =1).values
            network_names = already_created

        self.pkl = {'predictions': predictions,
                    'targets': targets,
                    'names': network_names}


    def score_metrics(self, confusion_plot_name = None, show_plot = True):
        # calculates confusion matrix, weighted accuracy, accuracy and AUC for given pkl
        if confusion_plot_name is not None:
            self.confusion_matrix_name = confusion_plot_name

        if type(self.pkl['predictions']) is not list:
            self.weighted_accuracy = metrics.balanced_accuracy_score(np.argmax(self.pkl['targets'],1),
                                                                     np.argmax(self.pkl['predictions'],1))

            self.accuracy = metrics.accuracy_score(np.argmax(self.pkl['targets'],1),
                                                   np.argmax(self.pkl['predictions'],1))

            self.confusion_matrix = metrics.confusion_matrix(np.argmax(self.pkl['targets'],1),
                                                                   np.argmax(self.pkl['predictions'],1))
            self.per_class = self.confusion_matrix.diagonal()/np.sum(self.confusion_matrix,axis = 1)
            self.results = {'ensemble_results': {
                'weighted_accuracy': self.weighted_accuracy,
                'accuracy': self.accuracy,
                'confusion_matrix': self.confusion_matrix,
                'per_class': self.per_class
            }}
            if confusion_plot_name is None:
                self.make_confusion_matrix(cf = self.confusion_matrix,categories=self.labels,cbar = False, figsize=(10,8))
            else:
                self.make_confusion_matrix(cf=self.confusion_matrix, categories=self.labels, cbar=False,show=False,
                                           figsize=(10, 8))

        else:
            self.weighted_accuracy = [metrics.balanced_accuracy_score(np.argmax(preds,1), np.argmax(tags,1))
                                      for preds, tags in zip(self.pkl['predictions'],self.pkl['targets'])]

            self.accuracy = [metrics.accuracy_score(np.argmax(preds, 1), np.argmax(tags, 1))
                                      for preds, tags in zip(self.pkl['predictions'], self.pkl['targets'])]

            self.confusion_matrix = [metrics.confusion_matrix(np.argmax(preds, 1), np.argmax(tags, 1))
                                      for preds, tags in zip(self.pkl['predictions'], self.pkl['targets'])]
            self.make_roc_curve()

            self.results = {}
            self.results['weighted_accuracy'] = {}
            self.results['accuracy'] = {}
            self.results['confusion_matrix'] = {}
            for idx, networks in enumerate(self.labels):
                self.results['weighted_accuracy'][networks] = self.weighted_accuracy[idx]
                self.results['accuracy'][networks] = self.accuracy[idx]
                self.results['confusion_matrix'][networks] = self.confusion_matrix[idx]

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
                              show = True
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
            for i in range(cf.shape[0]):
                for j in range(cf.shape[1]):
                    if i != j:
                        group_percentages.append('')
                    else:
                        try:
                            group_percentages.append("{0:.2%}".format(weighted[count]))
                            count += 1
                        except:
                            breakpoint()

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
        sns.set(font_scale = 1.2)
        sns.heatmap(cf, annot=box_labels, fmt="", cmap=cmap, cbar=cbar, xticklabels=categories, yticklabels=categories)

        if xyplotlabels:
            plt.ylabel('True label',size = 20)
            plt.xlabel('Predicted label', size = 20)
        else:
            plt.xlabel(stats_text)

        if title:
            plt.title(title)
        if show:
            # plt.rc('axes', label_size='10')
            plt.show()
        else:
            # plt.rc('axes', label_size = '10')
            plt.savefig(os.path.join(self.ground_path,self.confusion_matrix_name+'.eps'),format = 'eps',pad_inches = 0)

def ensembling(predictions_list,indices, labels, test_predictions = None, n_predictors = 'full', over_sampling = 0):
    """

    :param predictions_list: list of data_visualiser instances
    :param indices: train and validation indices
    :param labels: true labels of predictions
    :param test_predictions: list of data_visualiser instances on test set
    :return: new predictions
    """
    if n_predictors == 'full':
        x_matrix = np.zeros((len(labels), len(labels.drop(['image'],axis=1).columns) * len(predictions_list)))
    else:
        x_matrix = np.zeros((len(labels),len(predictions_list)))
    y_matrix = labels.drop(['image'], axis = 1).values
    for j, (networks_train, networks_val) in enumerate(predictions_list):
        if n_predictors == 'full':
            x_mat = np.zeros((y_matrix.shape[0]+over_sampling, 1))
            x_mat[indices['trainIndCV']] = networks_train.pkl['predictions']
            x_mat[indices['valIndCV']] = networks_val.pkl['predictions']

            if over_sampling != 0:
                max_index = np.max(list(set(indices['trainIndCV'])-set(indices['valIndCV'])))
                x_mat[max_index:] = np.random.choice(
                    np.argmax(networks_val.pkl['predictions'], axis = 1),
                    size = over_sampling)



        else:
            x_mat = np.zeros((y_matrix.shape[0]+over_sampling, 1))
            x_mat[indices['trainIndCV']] = np.argmax(networks_train.pkl['predictions'], axis = 1)
            x_mat[indices['valIndCV']] = np.argmax(networks_val.pkl['predictions'],axis = 1)

            if over_sampling != 0:
                max_index = np.max(list(set(indices['trainIndCV'])+set(indices['valIndCV'])))
                x_mat[max_index:] = np.random.choice(np.argmax(networks_val.pkl['predictions'],axis=1),
                                                     size = over_sampling)

        if j == 0:
            x_matrix = x_mat
        else:
            x_matrix = np.concatenate((x_matrix, x_mat), axis = 1)

    random_forest = RandomForestClassifier(n_estimators=100).fit(x_matrix,y_matrix).predict_proba(test_predictions)
    svc = SVC().fit(x_matrix,y_matrix).predict(test_predictions)
    gradient_boost = GradientBoostingClassifier().fit(x_matrix,y_matrix).predict_proba(test_predictions)
    decision_tree = DecisionTreeClassifier().fit(x_matrix,y_matrix).predict_proba(test_predictions)

    return random_forest, svc, gradient_boost, decision_tree



def main():
    our_ensemble = ['res101_rr', 'efficientnet_b5_rr', 'se_resnet101_rr', 'nasnetamobile_rr',
                    'efficientnet_b6_ss', 'resnext101_32_8_rr']

    their_ensemble = ['efficientnet_b0_rr','efficientnet_b1_rr', 'efficientnet_b2_rr', 'efficientnet_b3_rr',
                      'efficientnet_b4_rr', 'efficientnet_b5_rr', 'efficientnet_b0_ss', 'efficientnet_b1_ss',
                      'efficientnet_b2_ss', 'efficientnet_b3_ss', 'efficientnet_b4_ss', 'efficientnet_b5_ss',
                      'efficientnet_b6_ss', 'senet154_ss']

    test_res101 = ['res101']

    gustav_bunder = data_visualiser()
    # gustav_bunder.create_prediction_pkl(location= r'C:\Users\ptrkm\Bachelor\2018_test\eval_predictions_2018\2018_mixed.efficientnet_b4_rr_bestgpu2_90_predn.pkl',
    #                                     evaluation_set='AISC', )
    # gustav_bunder.create_pkl_new(network_names=test_res101, path_to_files=r'C:\Users\ptrkm\Bachelor\Eval predictions\eval_predictions_ISIC_AISC_on_ISIC2019',
    #                              prefix='/home/s184400/isic2019/ISIC_test_cropped/',create_test=True,name='2019_eval_trained_res',average_voting=True)

    gustav_bunder.evaluate_new(network_names=their_ensemble,path_to_files=r'C:\Users\ptrkm\Bachelor\Eval predictions\eval_predictions_2019_on_AISC',
                               prefix='/home/s184400/isic2019/AISC_images/official/',path_to_labels=r'C:\Users\ptrkm\Bachelor\labels.csv',
                               average_voting=True)
    peter_bunder_ikke = data_visualiser()
    peter_bunder_ikke.evaluate_new(network_names=our_ensemble,path_to_files=r'C:\Users\ptrkm\Bachelor\Eval predictions\eval_predictions_2019_on_AISC',
                               prefix='/home/s184400/isic2019/AISC_images/official/',path_to_labels=r'C:\Users\ptrkm\Bachelor\labels.csv',
                               average_voting=True)

    # gustav_bunder.evaluate_new_white(network_names=test_res101,path_to_files=r'C:\Users\ptrkm\Bachelor\Eval predictions\eval_predictions_wb2',
    #                                  prefix='/scratch/s184400/test_a/',emulator='mohan',path_to_labels=r'C:\Users\ptrkm\Bachelor\labels.csv')

    # gustav_bunder.create_pkl(already_created='their_ensemble')


    # gustav_bunder.create_predictixon_pkl(location='2019_rr.resnet101_rr_AISC_gpu0_58_predn.pkl')r'C:\Users\ptrkm\Bachelor\Eval predictions\eval_predictions_2019_on_AISC'
    gustav_bunder.score_metrics()
    peter_bunder_ikke.score_metrics()
    labels = pd.read_csv(r'C:\Users\ptrkm\Bachelor\labels.csv')


    pcl = {'their_ensemble':gustav_bunder.pkl,
           'our_ensemble':peter_bunder_ikke.pkl,
           'ground_truth_AISC':labels}

    with open(r'C:\Users\ptrkm\Bachelor\new_pickle_results.pkl','wb') as handle:
        pickle.dump(pcl,handle, protocol=pickle.HIGHEST_PROTOCOL)



    breakpoint()
main()