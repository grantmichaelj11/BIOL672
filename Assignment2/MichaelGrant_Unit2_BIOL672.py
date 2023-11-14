# -*- coding: utf-8 -*-
"""
@author: Michael J. Grant

Operating System: Windows 10/11
Packages: sys,numpy, pandas, sklearn, statsmodels, matplotlib, random, plotly
Data Files: NBA_gammas.csv

Important note about the dataset used:

For this unit assignment I will be using a self-procured dataset of NBA game statistics
from 2014-2018. These boxscores were scraped ethically from https://www.basketball-reference.com/
The database is stored locally and for purposes of this exercise I have generated
CSVs of desired metrics.

The purpose of this dataset is for predicting probabilites of a team winning to 
cover a certain sportbook's spread. Therefore, a n-day moving average was applied
to each statistic prior to the game to assess a team's momentum in each metric. 

The philosophy of this dataset is that in any given NBA game each team has 240
minutes of player time to win. While points per game is a simple measure I believe
it discredits other facets of the game such as defensive metrics. Therefore I have
engineered a feature I declare the "gamma" parameter.

For any given team gamma is given as a summation of the n-day moving averages:

       Stat                     Weight
Three Pointers Made                3
Field Goals Made                   2       
Free Throws Made                   1
Offensive Rebounds                 1 
Defensive Rebounds                 1
Steals                             1
Blocks                            0.5
Assists                           0.5
Turnovers                         -1
Field Goals Missed                -2
Three Pointers Missed             -3
Free Throws Missed                -1

These weights are arbitrary and model performance most likely could be improved
via an optimization. Gamma was calculated for both the home team and the away team.
"""

import sys
import pandas as pd
from statsmodels.multivariate.manova import MANOVA
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

# Imports from sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def main():
    
    df = test_gammas_across_seaons('NBA_gammas.csv', 'MANOVA_NBA_gammas.txt', plot=True)
    
    # Question 1 models
    knn_model(df, 'KNN_analysis.txt', plot=True)
    naive_bayes_model(df, 'Naive_Bayes_analysis.txt', plot=True)
    lda_model(df, 'LDA_analysis.txt', plot=True)
    qda_model(df, 'QDA_analysis.txt', plot=True)
    
    # Question 2 models
    linear_svm_model(df, 'Linear_SVM_analysis.txt', plot=True)
    quadratic_svm_model(df, 'Quadratic_SVM_analysis.txt', plot=True)
    rbf_svm_model(df, 'RBF_SVM_analysis.txt', plot=True)
    
    # Model Analysis for questions 1 and 2
    
    """ 
    Model                 Accuracy
    
    KNN                     71%
    Naive Bayes             73%
    LDA                     75%
    QDA                     74%
    
    Linear SVM              74%
    Polynomial SVM          74%
    RBF SVM                 74%
    
    The best performing model was the Linear Discriminant Analysis, but all models
    perform relatively well. If we wanted to blindly trust a model to predict if
    the home team was going to win the game or not we could use LDA. It is both
    time efficent has ~equal precision and recall, meaning the false positive and
    false negative rates are roughly the same in the prediction.
    
    KNN performed the worst.
    
    Precision = True Positives / (True Positives + False Positives)
    Recall = True Positives / (True Positives + False Negatives)

    In other words, precision is our ability to guess correctly weighted by 
    how many times we guessed the outcome but were wrong. Recall is our ability to
    guess correctly weighted by how many times we missed a proper prediction.
    
    KNN Precision and Recall
    
    Home Team Loss: Precision = 0.68, Recall = 0.78
    Home Team Wins: Precision = 0.75, Recall = 0.63
    
    Therefore the KNN  biases itself towards the home team losing. A lower
    precision with predicting losses shows we predicted more losses that were incorrect
    than wins. The recall being 0.78 shows that the number of times the home
    team was predicted to win but actually lost is lower than the number of times 
    they were predicted to lose but actually won. Therefore we should trust the KNN
    prediction more if it predicts the home team to win.
    
    The other model with drastically different precisions and recalls is the
    quadratic SVM model. Interestingly this model has the opposite bias of the
    KNN model in that it wants to predict that the home team will win.
    
    Home Team Loss: Precision = 0.78, Recall = 0.63
    Home Team Wins: Precision = 0.69, Recall = 0.82
    
    If we apply the same logic as before, we see that given a prediction that the 
    home team loses results in a smaller amount of relative false positives than
    if the model predicts a win. 
    
    The rest of the models have relatively unremarkable differences in recall and
    precision. 
    
    We could combine the logic of multiple models to further enhance out ability
    to make predictions. For example, we know that KNN wants to strongly favor
    the home team losing, the QSVM wants to favor the home team winning and our
    best model, the LDA is relatively agnostic, but has the best predicitve
    capability. Some scenarios exists that could shape our prediction
    
    Of note:
        LDA predictions are non-biased
        KNN biases home team losing (predictions of wins are welcome)
        QSVM biases home team winning (predictions of losses are welcome)
    
    Scenario 1: LDA predicts a win, QSVM predicts a win and KNN predicts a win
    
    In this scenario we know that QSVM wants to bias towards the home team winning
    and predicts a win, which is insignificnt. Yet, the KNN wants to bias the
    home team losing and we see it still predicts a win. Given this is opposite
    its bias and that LDA predicts a win, could this enhance the probability
    that we are correct in our prediction that the home team will win?
    
    This question I will answer in question number three. I will train two
    neural networks. One where the input parameters are simpy the gamma values 
    for the home and away teams, and the other will be the predictions of the
    LDA/KNN/QSVM models
    """
    
    # Question 3
    

def test_gammas_across_seaons(dataset, outputfile, plot=True):
    
    # Prior to any data manipulation I want to show that for all 4 seasons that we
    # can combine the data from all 4 seasons into one master dataset, and that 
    # gamma does not vary much from season to season.
    
    
    #Open dataset as a pandas dataframe
    gammas = pd.read_csv(dataset)
    gamma_distributions_per_season = [gammas[gammas['season']==2015]['gamma'],
                                      gammas[gammas['season']==2016]['gamma'],
                                      gammas[gammas['season']==2017]['gamma'],
                                      gammas[gammas['season']==2018]['gamma']]
    
    
    gammas_MANOVA = MANOVA.from_formula('gamma + opp_gamma ~ season', data=gammas)
    
    #Save output
    with open(outputfile, 'w') as f:
        original_stdout = sys.stdout
        sys.stdout = f
        print(gammas_MANOVA.mv_test())
        sys.stdout = original_stdout
        
        
    #Plot boxplot
    if plot == True:
        ax = plt.figure().add_subplot(111)
        
        plt.xlabel('Seasons')
        plt.ylabel('Gamma')
        
        ax.boxplot(gamma_distributions_per_season, labels=[2015,2016,2017,2018], showmeans=True)
        
        plt.savefig('boxplot_for_MANOVA_analysis.png')
        
        plt.show()
    
    return(gammas)

    # Visually we see that the mean and variance hardly changes from season to
    # season. This means we can combine all four seasons into our analysis as
    # no significant deviation is observed. This is further verified through the
    # MANOVA, where we see F-statistics of less than 1.0 with p-values of ~0.7.
    # This indicates that any difference in variance is due to randomness/chance.
    
def knn_model(df, outputfile, plot=True):
    
    gammas = df[['gamma', 'opp_gamma']].values
    win_loss = df['Winner'].values
    
    # split the data into training and test data - 50% of the data will be used
    # for training and 50% will be used to test the model
    gamma_train, gamma_test, win_prediction_train, win_prediction_test = train_test_split(gammas, win_loss, test_size=0.5)
    
    knn_classifier = KNeighborsClassifier(n_neighbors = 8)
    knn_classifier.fit(gamma_train, win_prediction_train)
    
    predictions = knn_classifier.predict(gamma_test)

    report = classification_report(win_prediction_test, predictions)
    
    with open(outputfile, 'w') as f:
        original_stdout = sys.stdout
        sys.stdout = f
        print(report)
        sys.stdout = original_stdout
    
    if plot == True:
        
        ax = plot_decision_regions(gamma_test, win_prediction_test, clf=knn_classifier)
        
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles,
                  ['Home Team Wins', 'Home Team Loses'])
        
        plt.xlabel('Away Team Gamma')
        plt.ylabel('Home Team Gamma')
        plt.title('K-Nearest Neighbor (n=8)')
        
        plt.savefig('Scatter_plot_decision_boundary_KNN.png')
        
        plt.show()
    
def naive_bayes_model(df, outputfile, plot=True):
    
    gammas = df[['gamma', 'opp_gamma']].values
    win_loss = df['Winner'].values
    
    # split the data into training and test data - 50% of the data will be used
    # for training and 50% will be used to test the model
    gamma_train, gamma_test, win_prediction_train, win_prediction_test = train_test_split(gammas, win_loss, test_size=0.5)
    
    NB_classifier = GaussianNB()
    NB_classifier.fit(gamma_train, win_prediction_train)
    
    predictions = NB_classifier.predict(gamma_test)

    report = classification_report(win_prediction_test, predictions)
    
    with open(outputfile, 'w') as f:
        original_stdout = sys.stdout
        sys.stdout = f
        print(report)
        sys.stdout = original_stdout
    
    if plot == True:
        
        ax = plot_decision_regions(gamma_test, win_prediction_test, clf=NB_classifier)
        
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles,
                  ['Home Team Wins', 'Home Team Loses'])
        
        plt.xlabel('Away Team Gamma')
        plt.ylabel('Home Team Gamma')
        plt.title('Gaussian Naive Bayes')
        
        plt.savefig('Scatter_plot_decision_boundary_naive_bayes.png')
        
        plt.show()

def lda_model(df, outputfile, plot=True):
    
    gammas = df[['gamma', 'opp_gamma']].values
    win_loss = df['Winner'].values
    
    # split the data into training and test data - 50% of the data will be used
    # for training and 50% will be used to test the model
    gamma_train, gamma_test, win_prediction_train, win_prediction_test = train_test_split(gammas, win_loss, test_size=0.5)
    
    LDA_classifier = LinearDiscriminantAnalysis()
    LDA_classifier.fit(gamma_train, win_prediction_train)
    
    predictions = LDA_classifier.predict(gamma_test)

    report = classification_report(win_prediction_test, predictions)
    
    with open(outputfile, 'w') as f:
        original_stdout = sys.stdout
        sys.stdout = f
        print(report)
        sys.stdout = original_stdout
    
    if plot == True:
        
        ax = plot_decision_regions(gamma_test, win_prediction_test, clf=LDA_classifier)
        
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles,
                  ['Home Team Wins', 'Home Team Loses'])
        
        plt.xlabel('Away Team Gamma')
        plt.ylabel('Home Team Gamma')
        plt.title('LDA Classification')
        
        plt.savefig('Scatter_plot_decision_boundary_LDA.png')
        
        plt.show()

def qda_model(df, outputfile, plot=True):
    
    
    gammas = df[['gamma', 'opp_gamma']].values
    win_loss = df['Winner'].values
    
    # split the data into training and test data - 50% of the data will be used
    # for training and 50% will be used to test the model
    gamma_train, gamma_test, win_prediction_train, win_prediction_test = train_test_split(gammas, win_loss, test_size=0.5)
    
    QDA_classifier = QuadraticDiscriminantAnalysis()
    QDA_classifier.fit(gamma_train, win_prediction_train)
    
    predictions = QDA_classifier.predict(gamma_test)

    report = classification_report(win_prediction_test, predictions)
    
    with open(outputfile, 'w') as f:
        original_stdout = sys.stdout
        sys.stdout = f
        print(report)
        sys.stdout = original_stdout
    
    if plot == True:
        
        ax = plot_decision_regions(gamma_test, win_prediction_test, clf=QDA_classifier)
        
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles,
                  ['Home Team Wins', 'Home Team Loses'])
        
        plt.xlabel('Away Team Gamma')
        plt.ylabel('Home Team Gamma')
        plt.title('QDA Classification')
        
        plt.savefig('Scatter_plot_decision_boundary_QDA.png')
        
        plt.show()
    
    
def linear_svm_model(df, outputfile, plot=True):
    
    gammas = df[['gamma', 'opp_gamma']].values
    win_loss = df['Winner'].values
    
    # split the data into training and test data - 50% of the data will be used
    # for training and 50% will be used to test the model
    gamma_train, gamma_test, win_prediction_train, win_prediction_test = train_test_split(gammas, win_loss, test_size=0.5)
    
    #For support vector machines it is important we standardize the features
    scaler = StandardScaler()
    gamma_train = scaler.fit_transform(gamma_train)
    gamma_test = scaler.transform(gamma_test)
    
    
    SVM_classifier = SVC(kernel='linear', C=1.0)
    SVM_classifier.fit(gamma_train, win_prediction_train)
    
    predictions = SVM_classifier.predict(gamma_test)

    report = classification_report(win_prediction_test, predictions)
    
    with open(outputfile, 'w') as f:
        original_stdout = sys.stdout
        sys.stdout = f
        print(report)
        sys.stdout = original_stdout
    
    if plot == True:
        
        ax = plot_decision_regions(gamma_test, win_prediction_test, clf=SVM_classifier)
        
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles,
                  ['Home Team Wins', 'Home Team Loses'])
        
        plt.xlabel('Away Team Gamma')
        plt.ylabel('Home Team Gamma')
        plt.title('Linear SVM Classification')
        
        plt.savefig('Scatter_plot_linear_SVM.png')
        
        plt.show()
        
def quadratic_svm_model(df, outputfile, plot=True):
    
    gammas = df[['gamma', 'opp_gamma']].values
    win_loss = df['Winner'].values
    
    # split the data into training and test data - 50% of the data will be used
    # for training and 50% will be used to test the model
    gamma_train, gamma_test, win_prediction_train, win_prediction_test = train_test_split(gammas, win_loss, test_size=0.5)
    
    #For support vector machines it is important we standardize the features
    scaler = StandardScaler()
    gamma_train = scaler.fit_transform(gamma_train)
    gamma_test = scaler.transform(gamma_test)
    
    
    SVM_classifier = SVC(kernel='poly', C=1.0)
    SVM_classifier.fit(gamma_train, win_prediction_train)
    
    predictions = SVM_classifier.predict(gamma_test)

    report = classification_report(win_prediction_test, predictions)
    
    with open(outputfile, 'w') as f:
        original_stdout = sys.stdout
        sys.stdout = f
        print(report)
        sys.stdout = original_stdout
    
    if plot == True:
        
        ax = plot_decision_regions(gamma_test, win_prediction_test, clf=SVM_classifier)
        
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles,
                  ['Home Team Wins', 'Home Team Loses'])
        
        plt.xlabel('Away Team Gamma')
        plt.ylabel('Home Team Gamma')
        plt.title('Quadratic SVM Classification')
        
        plt.savefig('Scatter_plot_quadratic_SVM.png')
        
        plt.show()
        
def rbf_svm_model(df, outputfile, plot=True):
    
    gammas = df[['gamma', 'opp_gamma']].values
    win_loss = df['Winner'].values
    
    # split the data into training and test data - 50% of the data will be used
    # for training and 50% will be used to test the model
    gamma_train, gamma_test, win_prediction_train, win_prediction_test = train_test_split(gammas, win_loss, test_size=0.5)
    
    #For support vector machines it is important we standardize the features
    scaler = StandardScaler()
    gamma_train = scaler.fit_transform(gamma_train)
    gamma_test = scaler.transform(gamma_test)
    
    
    SVM_classifier = SVC(kernel='rbf', C=1.0)
    SVM_classifier.fit(gamma_train, win_prediction_train)
    
    predictions = SVM_classifier.predict(gamma_test)

    report = classification_report(win_prediction_test, predictions)
    
    with open(outputfile, 'w') as f:
        original_stdout = sys.stdout
        sys.stdout = f
        print(report)
        sys.stdout = original_stdout
    
    if plot == True:
        
        ax = plot_decision_regions(gamma_test, win_prediction_test, clf=SVM_classifier)
        
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles,
                  ['Home Team Wins', 'Home Team Loses'])
        
        plt.xlabel('Away Team Gamma')
        plt.ylabel('Home Team Gamma')
        plt.title('RBF SVM Classification')
        
        plt.savefig('Scatter_plot_rbg_SVM.png')
        
        plt.show()

    
main()

























