from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

def precision_recall_curve(y, y_predict_proba):
    precisions, recalls, thresholds = precision_recall_curve(y, y_predict_proba)
    fig, ax = plt.subplots()
    ax.plot(thresholds, precisions[: -1]), "b--", label="Precision")
    ax.plot(thresholds, recalls[:-1], "g-", label='Recall')
    ax.set_xticks(thresholds)
    # add x labels
    # add y ticks & labeles
    # add grid

    ax.set_title('Precision and recall versus the decision threshold')

