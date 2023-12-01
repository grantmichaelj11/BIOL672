# -*- coding: utf-8 -*-
"""
@author: Grant

Operating System: Windows 10/11
Packages: sys,pandas, sklearn, MichaelGrant_Unit2_BIOL672
Data Files: NBA_gammas.csv
"""
import sys
import pandas as pd
from MichaelGrant_Unit2_BIOL672 import test_gammas_across_seaons
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

def main():
    """
    I want to know train a neural network based on the inputs from the LDA/KNN/QSVM models

    Rather than just take the binary output from the prediction as the inputs for the
    input layer of the neural network, I will instead use the probabilities of the home
    team winning to calculate these. I want to use probabilities instead of binary outputs 
    because this will accomodate more variance. For example, if a team has a 51%
    probabilitiy of winning in one game, and in another game they have a 90% chance of winning
    the binary output will be "1" in both scenarios. Both those possible outcomes should
    not be weighted the same.

    Two ANN were investigated and compared to the previous classification techniques
    The first network under consideration is using the gamma inputs that were used in
    the previous classifiers (home and opponent gammas). This model has a prediction 
    accuracy of 0.73 with relatively consistent precision and recalls. Therefore simply
    using the neural network did not perform better than the other models.

    Yet, when we took the previous classifiers with bias as already discussed, and used
    the probabilities from each model on the prediction of the home team winning, we observe
    an accuracy score of 0.76, which is 1% higher than the LDA. Therefore combining these
    methods slightly optimized our predicitive capabilities.

    The random forest analysis actually preformed the worst out of all the models 
    investigated with an accuracy rating of 69% but this was accompanied by consistent
    precision and recall. Therefore we will see less false negatives/positives when compared
    to some of the other classifiers (KNN for example).

    The gradient boosting analysis provided an accuracy rating of 73%. with identical
    recall and precision. While this model still does not perform better than 
    discriminant analysis or the support vector machines, it performs better than
    the random foreset classifier, which is another ensemble technique.

    Further optimization could be accomplished by tuning the hyperparameters for each ensemble, 
    especially gradient boosting as it is more prone to overfitting.

    The training/test split was 50:50 for cross validation. Therefore we can generally
    assume that we did not overfit our models. Further analysis such as k-cross-fold 
    validation could further prove this point.
    """
    
    
    raw_NN = NNClassifier_gammas("NBA_gammas.csv", 2, 64, "classifier_analysis/NeuralNetwork_Gammas_Analysis.txt")
    
    combined_NN = NNClassifier_combined_models(2,64,"classifier_analysis/NeuralNetwork_Combined_Analysis.txt.")
    
    random_forest = RandomForest_Classifier("NBA_gammas.csv", "classifier_analysis/RandomForest_Analysis.txt")
    
    gradient_boosting = BoostedGradient_Classifier("NBA_gammas.csv", "classifier_analysis/GradientBoosting_Analysis.txt")

# Create the dataset needed for the neural network.
#LDA Prob | KNN Prob | QSVM Prob | Win/Loss
def generate_probabilites_dataset():
    
    #Load in dataset    
    df = test_gammas_across_seaons('NBA_gammas.csv', plot=False)

    gammas = df[['gamma', 'opp_gamma']].values
    win_loss = df['Winner'].values
    
    #Normalize Data
    scaler = StandardScaler()
    gamma_train = scaler.fit_transform(gammas)
    
    knn_classifier = KNeighborsClassifier(n_neighbors = 8)
    knn_classifier.fit(gamma_train, win_loss)
    knn_probabilities = knn_classifier.predict_proba(gamma_train)[:,1]
    
    LDA_classifier = LinearDiscriminantAnalysis()
    LDA_classifier.fit(gamma_train, win_loss)
    LDA_probabilities = LDA_classifier.predict_proba(gamma_train)[:,1]
    
    SVM_classifier = SVC(kernel='poly', C=1.0, probability=True)
    SVM_classifier.fit(gamma_train, win_loss)
    SVM_probabilities = SVM_classifier.predict_proba(gamma_train)[:,1]
    
    #We have the probabilities for the home team winning, and now we need to create
    #Our dataframe
    
    df_dictionary = {"LDA": LDA_probabilities,
                     "KNN": knn_probabilities,
                     "SVM": SVM_probabilities,
                     "Win": win_loss}
    
    return pd.DataFrame(df_dictionary)
  
def NNClassifier_combined_models(layers, neurons, outputfile=""):

    #Generate Dataframe
    data = generate_probabilites_dataset()
    
    inputs = data.drop('Win',axis=1)
    outputs = data['Win']
    
    #We will keep the same training/test split of 0.5
    X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size=0.5)    

    clf = MLPClassifier(solver="adam",
                        hidden_layer_sizes=(layers,neurons),
                        shuffle = True
                        )
    
    model = clf.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    report = classification_report(y_test, y_pred)
    
    if outputfile != "":
        with open(outputfile, 'w') as f:
            original_stdout = sys.stdout
            sys.stdout = f
            print(report)
            sys.stdout = original_stdout
    
    return model

def NNClassifier_gammas(dataset, layers, neurons, outputfile=""):
    
    df = pd.read_csv(dataset)
    
    gammas = df[['gamma', 'opp_gamma']].values
    win_loss = df['Winner'].values
    
    gamma_train, gamma_test, win_prediction_train, win_prediction_test = train_test_split(gammas, win_loss, test_size=0.5)

    clf = MLPClassifier(solver='adam',
                        hidden_layer_sizes=(layers, neurons),
                        shuffle = True)
    
    model = clf.fit(gamma_train, win_prediction_train)
    
    y_pred = model.predict(gamma_test)
    
    report = classification_report(win_prediction_test, y_pred)
    
    if outputfile != "":
        with open(outputfile, 'w') as f:
            original_stdout = sys.stdout
            sys.stdout = f
            print(report)
            sys.stdout = original_stdout
            
    return model

def RandomForest_Classifier(dataset, outputfile=""):
    
    df = pd.read_csv(dataset)
    
    gammas = df[['gamma', 'opp_gamma']].values
    win_loss = df['Winner'].values
    
    gamma_train, gamma_test, win_prediction_train, win_prediction_test = train_test_split(gammas, win_loss, test_size=0.5)
    
    clf = RandomForestClassifier()
    
    model = clf.fit(gamma_train, win_prediction_train)
    
    y_pred = model.predict(gamma_test)
    
    report = classification_report(win_prediction_test, y_pred)
    
    if outputfile != "":
        with open(outputfile, 'w') as f:
            original_stdout = sys.stdout
            sys.stdout = f
            print(report)
            sys.stdout = original_stdout
            
    return model

def BoostedGradient_Classifier(dataset, outputfile=""):
    
    df = pd.read_csv(dataset)
    
    gammas = df[['gamma', 'opp_gamma']].values
    win_loss = df['Winner'].values
    
    gamma_train, gamma_test, win_prediction_train, win_prediction_test = train_test_split(gammas, win_loss, test_size=0.5)
    
    clf = GradientBoostingClassifier()
    
    model = clf.fit(gamma_train, win_prediction_train)
    
    y_pred = model.predict(gamma_test)
    
    report = classification_report(win_prediction_test, y_pred)
    
    if outputfile != "":
        with open(outputfile, 'w') as f:
            original_stdout = sys.stdout
            sys.stdout = f
            print(report)
            sys.stdout = original_stdout
            
    return model
    
x = main()
    
    
    
    
    
    
    
    
    
    