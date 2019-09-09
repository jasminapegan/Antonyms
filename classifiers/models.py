import numpy as np
from random import shuffle
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from get_data_array import get_data_array

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from joblib import dump, load

from keras.models import load_model, Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping

# example from https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html
# also from https://www.learnopencv.com/svm-using-scikit-learn-in-python/

def find_params_svm(train_data, kernel, C_range, gamma_or_degree_range, out_file, plot=False):
    """
        Finds best params for svm model, 'kernel' can be "rbf" or "poly".
        'C_range' and 'gamma_or_degree_range' are lists or other ranges to check.
        Optional parameter 'plot' sets if a plot of results should be drawn.
    """

    X, Y = get_data_array(train_data)

    ######################################################
    param_grid = dict(gamma=gamma_or_degree_range, C=C_range)
    grid = GridSearchCV(SVC(kernel=kernel), param_grid=param_grid, verbose=1)
    grid.fit(X, Y)

    with open(out_file, "a") as out:
        out.write("kernel: %s\n" % kernel)
        out.write("C range: %s\n" % str(C_range))
        if kernel == "svm":
            out.write("gamma range: %s\n" % str(gamma_or_degree_range))
        else:
            out.write("degree range: %s\n" % str(gamma_or_degree_range))
        out.write("Best Parameters: %s\n" % str(grid.best_params_))
        out.write("The best parameters are %s with a score of %0.4f\n" % (grid.best_params_, grid.best_score_))
        out.write(50 * "#" + "\n")

    if plot:
        ######################################################################
        # Utility function to move the midpoint of a colormap to be around
        # the values of interest.
        # based on https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html

        class MidpointNormalize(Normalize):

            def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
                self.midpoint = midpoint
                Normalize.__init__(self, vmin, vmax, clip)

            def __call__(self, value, clip=None):
                x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
                return np.ma.masked_array(np.interp(value, x, y))

        scores = grid.cv_results_['mean_test_score'].reshape(len(C_range),
                                                             len(gamma_or_degree_range))

        plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
                   norm=MidpointNormalize(vmin=0.2, midpoint=0.92))
        if kernel == "rbf":
            plt.xlabel('gamma')
        else:
            plt.xlabel('degree')
        plt.ylabel('C')
        plt.colorbar()
        plt.xticks(np.arange(len(gamma_or_degree_range)), gamma_or_degree_range, rotation=45)
        plt.yticks(np.arange(len(C_range)), C_range)
        plt.title('Validation accuracy')
        plt.show()


def find_params_neural(k_range, k2_scale, k3_scale, k4_scale, train_set, model_file, out_file,
                       d1=0.05, d2=0.2, epochs=1000):
    """
        Makes models with all combinations of given 'k_range', 'k2_scale', 'k3_scale', 'k4_scale' for its layer sizes.
        Ex. k_range=[64], k2_scale=[4], k3_scale=[2], k4_scale=[2] means layers with 64, 64/4=16, 16/2=8, 8/2=4 neurons.
        If k4_scale=[0], there will be only 2 hidden layers, if k3_scale=[0] we get 1 hidden layer,
        with k2_scale=[0] we don't have any hidden layer.
        Models are trained and tested on given 'train_set' and output is printed to 'out_file'.
        Optional parameters: 'd1'=0.05 is dropout of input and output layer, 'd2'=0.2 is dropout of hidden layers,
        'epochs'=1000 is number of learning epochs. Model stops learning earlier because patience is set to 50.

    """

    X, Y = get_data_array(train_set)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

    with open(out_file, "a") as out:
        out.write("epochs: %d\n" % epochs)

        # k, k2, k3, k4, epochs, d1, d2, testAcc
        best_result = [0, 0, 0, 0, 0, 0, 0, 0]

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)

        for k in k_range:
            for k2s in k2_scale:
                if k2s != 0:
                    k2 = k // k2s
                for k3s in k3_scale:
                    if k3s != 0:
                        k3 = k2 // k3s
                    for k4s in k4_scale:
                        if k4s != 0:
                            k4 = k3 // k4s

                        out.write("k: %d, k2: %d, k3: %d, k4: %d\n" % (k, k2, k3, k4))

                        # build model
                        model = Sequential()
                        model.add(Dense(k, activation='sigmoid'))
                        model.add(Dropout(d1))
                        if k2s != 0:
                            model.add(Dense(k2, activation='sigmoid'))
                            model.add(Dropout(d2))
                        if k3s != 0:
                            model.add(Dense(k3, activation='sigmoid'))
                            model.add(Dropout(d2))
                        if k4s != 0:
                            model.add(Dense(k4, activation='sigmoid'))
                            model.add(Dropout(d2))
                        model.add(Dense(1, activation='sigmoid'))
                        model.add(Dropout(d1))

                        # compile model
                        model.compile(loss='binary_crossentropy',
                                      optimizer='adadelta',
                                      metrics=['accuracy'])

                        # train and test model
                        model.fit(X_train, Y_train, epochs=epochs, batch_size=100,
                                  validation_split=0.2, verbose=1, callbacks=[es])
                        _, trainAcc = model.evaluate(X_train, Y_train, batch_size=100)
                        _, testAcc = model.evaluate(X_test, Y_test, batch_size=100)

                        out.write("train/test accuracy: %f, %f\n" % (trainAcc, testAcc))

                        if testAcc > best_result[7]:
                            model.save(model_file)
                            best_result = [k, k2, k3, k4, epochs, d1, d2, testAcc]

        out.write("\t\t\t\t[k, k2, k3, k4, epochs, d1, d2, testAcc]\n")
        out.write("best result: \t" + str(best_result) + "\n")
        out.write(50 * "#" + "\n")


def make_models_svm(train_set, params_list, kernel, out_file):
    """ Train and save SVM models with params in 'params_list' to "svm_[kernel]_[C]_[gamma_or_degree].joblib". """
    # example from https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html

    X, Y = get_data_array(train_set)

    with open(out_file, "a") as out:
        for C, gamma_or_degree in params_list:
            filename = "svm_%s_%d_%d.joblib" % (kernel, C, gamma_or_degree)
            out.write(50 * "#" + "\n")
            out.write(filename)

            if kernel == "rbf":
                out.write("C: %0.3f, gamma: %0.3f\n" % (C, gamma_or_degree))
            else:
                out.write("C: %0.3f, degree: %0.3f\n" % (C, gamma_or_degree))

            clf = SVC(kernel=kernel, C=C, gamma=gamma_or_degree)
            if kernel == "poly":
                clf = SVC(kernel=kernel, C=C, degree=gamma_or_degree)

            out.write("cross_val_score\n")
            scores = cross_val_score(clf, X, Y, cv=3)
            out.write(str(sum(scores) / len(scores)) + "\n")
            out.write(40 * "#")
            clf.fit(X, Y)
            dump(clf, filename)

            print(scores)


def make_model_neural(train_set, model_file, out_file, plot=False):
    X, Y = get_data_array(train_set)

    model = load_model(model_file)
    model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    # train and test model
    history = model.fit(X, Y, epochs=5,
                        batch_size=100, validation_split=0.2,
                        verbose=1)
    _, train_acc = model.evaluate(X, Y, batch_size=100)

    with open(out_file, "a") as out:
        out.write("train acc: %d" % train_acc)
    model.save(model_file)

    if plot:
        # Plot training & validation accuracy values
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.legend(["acc", "val_acc", "val_loss"], loc='center right')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.title("Model accuracy")
        plt.show()


def kfold(train_set, file_list, out_file):
    """
        Splits 'train_set' into equal 5 parts, evaluates models from 'file_list'
        using 5-fold cross-validation and prints classification accuracies in 'out_file'.
    """
    # using example from https://keras.io/

    X, Y = get_data_array(train_set)

    train_X, test_kfold_X, train_Y, test_kfold_Y = train_test_split(X, Y, test_size=0.2, random_state=1)
    size_quarter = len(test_kfold_X) - 1  # subtract 1 to avoid quarters of different sizes
    train_X_parts = [train_X[:size_quarter],
                   train_X[size_quarter: 2 * size_quarter],
                   train_X[2 * size_quarter: 3 * size_quarter],
                   train_X[3 * size_quarter: 4 * size_quarter],
                   test_kfold_X[:-1]]
    train_Y_parts = [train_Y[:size_quarter],
                   train_Y[size_quarter: 2 * size_quarter],
                   train_Y[2 * size_quarter: 3 * size_quarter],
                   train_Y[3 * size_quarter: 4 * size_quarter],
                   test_kfold_Y[:-1]]

    with open(out_file, "a") as file:
        for filename in file_list:
            file.write("\n" + 40 * "#")
            file.write("\nmodel: " + filename + "\n")

            # k-fold check
            for k in range(5):
                testing_X = train_X_parts[k]
                training_X = np.concatenate((train_X_parts[(k + 1) % 5], train_X_parts[(k + 2) % 5],
                                            train_X_parts[(k + 3) % 5], train_X_parts[(k + 4) % 5]))
                testing_Y = train_Y_parts[k]
                training_Y = np.concatenate((train_Y_parts[(k + 1) % 5], train_Y_parts[(k + 2) % 5],
                                            train_Y_parts[(k + 3) % 5], train_Y_parts[(k + 4) % 5]))

                if filename[-7:] == ".joblib":
                    # load model
                    model = load("models/" + filename)

                    # train and test model: scikit-learn
                    model.fit(training_X, training_Y)
                    predictions = model.predict(testing_X)
                    score = accuracy_score(predictions, testing_Y)
                    file.write("test acc: " + str(score) + "\n")

                elif filename[-3:] == ".h5":
                    model = load_model("models/" + filename)
                    # compile model
                    model.compile(loss='binary_crossentropy',
                                  optimizer='adadelta',
                                  metrics=['accuracy'])
                    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
                    model.fit(training_X, training_Y, epochs=1000, validation_split=0.2, callbacks=[es], batch_size=100)

                    # train and test model: Keras
                    _, train_acc = model.evaluate(testing_X, testing_Y, batch_size=100)
                    _, test_acc = model.evaluate(testing_X, testing_Y, batch_size=100)
                    res = 'Train: %.3f, Test: %.3f' % (train_acc, test_acc)
                    file.write(res + "\n")

