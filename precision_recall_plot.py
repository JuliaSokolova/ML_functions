from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import numpy as np

def precision_recall_curve(y, y_predict_proba):
    precisions, recalls, thresholds = precision_recall_curve(y, y_predict_proba)
    fig, ax = plt.subplots()
    ax.plot(thresholds, precisions[: -1], "b--", label="Precision")
    ax.plot(thresholds, recalls[:-1], "g-", label='Recall')
    ax.set_xticks(thresholds)
    # ax.set_xticklabel(thresholds)
    y = np.arange(0, 1.1, 0.1)
    ax.set_ytics(y)
    ax.set_yticklabel(y)
    # add grid
    ax.grid(True)

    ax.set_title('Precision and recall versus the decision threshold')

    plt.show()

