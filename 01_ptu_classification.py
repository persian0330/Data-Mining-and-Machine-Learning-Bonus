"""
@author: K. Kersting, Z. Yu, J. Czech
Machine Learning Group, TU Darmstadt
"""
from sklearn import tree
from sklearn.model_selection import train_test_split
import numpy as np
import graphviz
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from util import get_x_down_sampled
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
import pydotplus
from collections import OrderedDict

def fit_dt_classifier(X_train: np.ndarray, y_train: np.ndarray, max_depth: int = None) -> tree.DecisionTreeClassifier:
    """Creates and fits a decision tree classifier on the training data."""
    
    # max_depth:The maximum depth of the tree
    # Use sklearn.tree.DecisionTreeClassifier()
    classifier=tree.DecisionTreeClassifier(max_depth=max_depth)
    
    # fit():Build a decision tree classifier from the training set(X_train, y_train)
    classifier=classifier.fit(X_train,y_train)
    
    return classifier


def get_test_accuracy(clf, X_test: np.ndarray, y_test: np.ndarray) -> float:
    """Evaluates the test accuracy for a given classifier and test dataset."""
    
    acc=clf.score(X_test,y_test)
    return acc


def export_tree_plot(clf, filename: str) -> None:
    """Exports the tree plot for the given classifier as a pdf with given filename."""
    
    # Export the decision tree classifier process as a graphviz.dot file.
    dot_data=tree.export_graphviz(clf)
    
    # A Dot class will be returned to represent the graph
    graph=pydotplus.graph_from_dot_data(dot_data)
    
    # Generate PDF file
    graph.write_pdf(filename)


def visualize_data(X_train: np.ndarray, y_train: np.ndarray, nb_samples_to_plot: int) -> None:
    """Visualizes the first nb_samples_to_plot in a plot with a legend of the class and exports the plot as a pdf."""

    # nb_samples_to_plot=10
    X_train_plot=X_train[:nb_samples_to_plot]
    y_train_plot=y_train[:nb_samples_to_plot]

    fig,ax=plt.subplots()
    t=np.linspace(0,5000,X_train.shape[1])
    # Xaxis:t, Yaxis:=X_train[:10], true_label:y_train[:10]
    for Y,true_label in zip(X_train_plot,y_train_plot):
        if true_label==100:
            ax.plot(t,Y,c='r',label='Speed=100')
        elif true_label==200:
            ax.plot(t,Y,c='g',label='Speed=200')
        else:
            ax.plot(t,Y,c='b',label='Speed=300')

    # Return handles and labels for legend
    handles,labels=ax.get_legend_handles_labels()
    
    # Avoid duplicate entries in the legend
    by_label=OrderedDict(zip(labels,handles))
    ax.legend(by_label.values(),by_label.keys(),loc='best')
    
    ax.set_title('first 10 time series of the training data')
    plt.xlabel("T",fontsize=20)
    plt.ylabel("Force",fontsize=20)
    plt.savefig('./speed_data.pdf')
    plt.show()


def run_grid_search(X_train, y_train, max_depth_range, learning_rate_range, n_estimators_range) -> (int, float, int):
    """Runs a grid search on the training data using the specified hyperparameter ranges using
     a 5 fold cross-valdidation per configuration. At last, the best hyperparamter tuple is returned."""

    df_clf=tree.DecisionTreeClassifier()
    
    parameters={'base_estimator__max_depth':max_depth_range,'learning_rate':learning_rate_range,'n_estimators':n_estimators_range}
    Ada_clf=AdaBoostClassifier(df_clf)
    
    # Exhaustive search over specified parameter values for an estimator.
    clf=GridSearchCV(Ada_clf,parameters,n_jobs=-1)
    clf.fit(X_train,y_train)
    
    return clf.best_params_['base_estimator__max_depth'], clf.best_params_['learning_rate'], clf.best_params_['n_estimators']


def fit_ada_boost(X_train: np.ndarray, y_train: np.ndarray, max_depth: int, learning_rate: float, n_estimators: int) -> AdaBoostClassifier:
    """Creates and fits an ada boost classifier on the training data."""
    # Use the best hyperparameters(max_depth,learning_rate,n_estimators) obtained through grid search
    df_clf=tree.DecisionTreeClassifier(max_depth=max_depth)
    
    Ada_clf=AdaBoostClassifier(base_estimator=df_clf,learning_rate=learning_rate,n_estimators=n_estimators)

    Ada_clf.fit(X_train,y_train)
    return Ada_clf


def main():
    # for reproducibility
    np.random.seed(42)

    # load data
    X_data = get_x_down_sampled('./PtU/FileName_Fz_raw.csv')
    y_data = np.loadtxt(open('./PtU/FileName_Speed.csv', 'r'), delimiter=",", skiprows=0)

    print("X_data.shape:", X_data.shape)
    print("y_data.shape:", y_data.shape)
    print("X_sample.shape:", X_data.shape)

    # split training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

    # visualize the task
    visualize_data(X_train, y_train, 10)

    # train
    clf = fit_dt_classifier(X_train, y_train, max_depth=3)
    # predict
    acc = get_test_accuracy(clf, X_test, y_test)
    print('Test Accuracy:', acc)

    print("predict_proba:", clf.predict_proba(X_test))

    # plot tree
    export_tree_plot(clf, "classification_tree.pdf")

    # run grid search
    max_depth_range = list(range(1, 5))
    learning_rate_range = [2 ** i for i in range(-2, 2)]
    n_estimators_range = [2 ** i for i in range(5, 8)]
    best_max_depth, best_lr, best_n_estimators = run_grid_search(X_train, y_train,  max_depth_range,
                                                                 learning_rate_range, n_estimators_range)
    clf = fit_ada_boost(X_train, y_train, best_max_depth, best_lr, best_n_estimators)
    print(f'Best configuration: max_depth: {best_max_depth}, lr: {best_lr}, n_estimators: {best_n_estimators}')

    # predict
    y_pred = clf.predict(X_test)
    print("y_pred:", y_pred[:10], "...")
    print("y_test:", y_test[:10], '...')

    # show confusion matrix
    print("Train Confusion Matrix for Ada Boost Classifier:\n", confusion_matrix(y_train, clf.predict(X_train)))
    print("Test Confusion Matrix for Ada Boost Classifier:\n", confusion_matrix(y_test, y_pred))

    # evaluate
    acc = accuracy_score(y_test, y_pred)
    print('Ada Boost Test Accuracy:', acc)


if __name__ == '__main__':
    main()
