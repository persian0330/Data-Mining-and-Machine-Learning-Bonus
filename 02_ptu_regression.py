"""
Created on January 25, 2018
@author: K. Kersting, Z. Yu, J. Czech
Machine Learning Group, TU Darmstadt
"""
from sklearn import tree
from sklearn.model_selection import train_test_split
import numpy as np
import graphviz
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from util import get_x_down_sampled
import pydotplus

def fit_dt_regressor(X_train: np.ndarray, y_train: np.ndarray, max_depth: int = None) -> tree.DecisionTreeRegressor:
    """Creates and fits a regression tree on the training data."""
    
    # max_depth:The maximum depth of the tree
    # Use sklearn.tree.DecisionTreeRegressor()
    regressor = tree.DecisionTreeRegressor(max_depth=max_depth)
    
    # fit():Build a decision tree regressor from the training set(X_train, y_train)
    regressor.fit(X_train,y_train)

    return regressor


def get_test_mse(clf, X_test: np.ndarray, y_test: np.ndarray) -> float:
    """Evaluates the test mse for a given classifier and test dataset."""

    y_test_pred = clf.predict(X_test)
    # Return mean squared error regression loss
    return mean_squared_error(y_test,y_test_pred)


def export_tree_plot(clf, filename: str) -> None:
    """Exports the tree plot for the given classifier as a pdf with given filename."""
    
    # Export the decision tree regressor process as a graphviz.dot file.
    dot_data = tree.export_graphviz(clf)
    
    # A Dot class will be returned to represent the graph
    graph = pydotplus.graph_from_dot_data(dot_data)
    
    # Generate PDF file
    graph.write_pdf(filename)


def visualize_thickness_data(X_train: np.ndarray, y_train: np.ndarray, nb_samples_to_plot: int) -> None:
    """Visualize the data by color encoding the thickness variable on a colormap and export the plot as pdf."""

    # print(X_train.shape)(2229,50)
    # print(y_train.shape)(2229,)
    
    # nb_samples_to_plot=10
    X_train_plot = X_train[:nb_samples_to_plot]
    y_train_plot = y_train[:nb_samples_to_plot]
    
    plt.figure()
    t = np.linspace(0, 5000, X_train.shape[1])
    # Xaxis:t, Yaxis:=X_train[:10], true_label:y_train[:10]
    for Y,true_label in zip(X_train_plot,y_train_plot):
        plt.plot(t,Y,c=plt.cm.inferno((true_label-np.min(y_train_plot))/(np.max(y_train_plot)-np.min(y_train_plot))))

    sm=plt.cm.ScalarMappable(cmap="inferno",norm=plt.Normalize(np.min(y_train_plot),np.max(y_train_plot)))
    cb=plt.colorbar(sm)
    cb.set_label("Thickness",fontsize=20)
    plt.xlabel("T",fontsize=20)
    plt.ylabel("Force",fontsize=20)
    plt.savefig('./thickness_data.pdf')
    plt.show()


def plot_thickness_histogram(y_train: np.ndarray, bins: 50) -> None:
    """Plots a histogram of 50 bins of the thickness distribution."""
    
    plt.figure()
    N,Bins,Patches=plt.hist(y_train,bins)
    for i in range(bins):
        Patches[i].set_facecolor(plt.cm.inferno(((Bins[i]+Bins[i+1])/2-np.min(y_train))/(np.max(y_train)-np.min(y_train))))
    plt.xlabel("Thickness",fontsize=20)
    plt.ylabel("Frequency",fontsize=20)
    sm=plt.cm.ScalarMappable(cmap="inferno",norm=plt.Normalize(np.min(y_train),np.max(y_train)))
    cb=plt.colorbar(sm)
    cb.set_label("Thickness",fontsize=20)
    plt.savefig('./thickness_histogram.pdf')
    plt.show()

def main():
    # for reproducibility
    np.random.seed(42)

    # load data
    X_data = get_x_down_sampled('./PtU/FileName_Fz_raw.csv')
    y_data = np.loadtxt(open('./PtU/FileName_thickness.csv', 'r'), delimiter=",", skiprows=0)
    print("X_data.shape:", X_data.shape)
    print("y_data.shape:", y_data.shape)

    # split training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

    # visualize the task
    visualize_thickness_data(X_train, y_train, 10)
    plot_thickness_histogram(y_train, 50)

    # train
    clf = fit_dt_regressor(X_train, y_train)
    # predict % evaluate
    mse = get_test_mse(clf, X_test, y_test)
    print('Test MSE:', mse)

    # change max tree depth
    # train
    clf = fit_dt_regressor(X_train, y_train, max_depth=3)

    # predict & evaluate
    mse = get_test_mse(clf, X_test, y_test)
    print('Test MSE:', mse)

    # plot tree
    export_tree_plot(clf, "regression_tree_d3.pdf")


if __name__ == '__main__':
    main()
