import matplotlib
import numpy as np
import data_parser
import matplotlib.pyplot as plt
from mean_error import mean_error
from sklearn import cross_validation
from sklearn.metrics import mean_squared_error


def execute(model, data, savepath, num_runs=200, num_folds=5, *args, **kwargs):
    responseSquared = True
    # get data
    Ydata = np.array(data.get_y_data()).ravel()
    Xdata = np.array(data.get_x_data())

    Y_predicted_best = []
    Y_predicted_worst = []

    maxRMS = -float('inf')
    minRMS = float('inf')

    RMS_List = []
    ME_List = []
    for n in range(num_runs):
        kf = cross_validation.KFold(len(Xdata), n_folds=num_folds, shuffle=True)
        K_fold_rms_list = []
        K_fold_me_list = []
        Overall_Y_Pred = np.zeros(len(Xdata))
        # split into testing and training sets
        for train_index, test_index in kf:
            X_train, X_test = Xdata[train_index], Xdata[test_index]
            Y_train, Y_test = Ydata[train_index], Ydata[test_index]
            # train on training sets
            model.fit(X_train, Y_train)
            Y_test_Pred = model.predict(X_test)
            rms = np.sqrt(mean_squared_error(Y_test, Y_test_Pred))
            me = mean_error(Y_test_Pred,Y_test)
            if responseSquared:
                Y_test = np.power(Y_test,0.5)
                Y_test_Pred[Y_test_Pred < 0] = 0
                Y_test_Pred = np.power(Y_test_Pred, 0.5)
                rms = np.sqrt(mean_squared_error(Y_test, Y_test_Pred))
                me = mean_error(Y_test_Pred, Y_test)
            K_fold_rms_list.append(rms)
            K_fold_me_list.append(me)
            Overall_Y_Pred[test_index] = Y_test_Pred

        RMS_List.append(np.mean(K_fold_rms_list))
        ME_List.append(np.mean(K_fold_me_list))
        if np.mean(K_fold_rms_list) > maxRMS:
            maxRMS = np.mean(K_fold_rms_list)
            Y_predicted_worst = Overall_Y_Pred

        if np.mean(K_fold_rms_list) < minRMS:
            minRMS = np.mean(K_fold_rms_list)
            Y_predicted_best = Overall_Y_Pred

    avgRMS = np.mean(RMS_List)
    medRMS = np.median(RMS_List)
    sd = np.std(RMS_List)
    meanME = np.mean(ME_List)

    print("Using {}x {}-Fold CV: ".format(num_runs, num_folds))
    print("The average RMSE was {:.3f}".format(avgRMS))
    print("The median RMSE was {:.3f}".format(medRMS))
    print("The max RMSE was {:.3f}".format(maxRMS))
    print("The min RMSE was {:.3f}".format(minRMS))
    print("The std deviation of the RMSE values was {:.3f}".format(sd))

    f, ax = plt.subplots(1, 2, figsize = (11,5))
    if responseSquared: ax[0].scatter(np.power(Ydata,0.5), Y_predicted_best, c='black', s=10)
    else: ax[0].scatter(Ydata, Y_predicted_best, c='black', s=10)
    ax[0].plot(ax[0].get_ylim(), ax[0].get_ylim(), ls="--", c=".3")
    ax[0].set_title('Best Fit')
    ax[0].text(.05, .88, 'Min RMSE: {:.2f} MPa'.format(minRMS), transform=ax[0].transAxes)
    ax[0].text(.05, .81, 'Mean RMSE: {:.2f} MPa'.format(avgRMS), transform=ax[0].transAxes)
    ax[0].text(.05, .74, 'Std. Dev.: {:.2f} MPa'.format(sd), transform=ax[0].transAxes)
    ax[0].text(.05, .67, 'Mean Mean Error.: {:.2f} MPa'.format(meanME), transform=ax[0].transAxes)
    ax[0].set_xlabel('Measured (Mpa)')
    ax[0].set_ylabel('Predicted (Mpa)')

    if responseSquared: ax[1].scatter(np.power(Ydata,0.5), Y_predicted_worst, c='black', s=10)
    else: ax[1].scatter(Ydata, Y_predicted_worst, c='black', s=10)
    ax[1].plot(ax[1].get_ylim(), ax[1].get_ylim(), ls="--", c=".3")
    ax[1].set_title('Worst Fit')
    ax[1].text(.05, .88, 'Max RMSE: {:.3f}'.format(maxRMS), transform=ax[1].transAxes)
    ax[1].set_xlabel('Measured (Mpa)')
    ax[1].set_ylabel('Predicted (Mpa)')

    f.tight_layout()
    f.savefig(savepath.format("cv_best_worst"), dpi=200, bbox_inches='tight')
    plt.clf()
    plt.close()
