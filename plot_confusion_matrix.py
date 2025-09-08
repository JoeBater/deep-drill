
def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True,
                          size=(8,6)):
    """
    Diagnostic confusion matrix plot with safeguards against zero-row normalization errors.
    """

    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    # Compute accuracy and misclassification
    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    # Default colormap
    if cmap is None:
        cmap = plt.get_cmap('Blues')

    # Normalize safely
    if normalize:
        row_sums = cm.sum(axis=1)
        zero_rows = np.where(row_sums == 0)[0]
        if len(zero_rows) > 0:
            print(f"⚠️ Warning: No true instances for classes at rows {zero_rows}")
            row_sums[zero_rows] = 1  # prevent division by zero
        cm = cm.astype('float') / row_sums[:, np.newaxis]

    # Plot setup
    plt.figure(figsize=size)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    # Tick labels
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    # Cell annotations
    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        value = "{:0.4f}".format(cm[i, j]) if normalize else "{:,}".format(cm[i, j])
        plt.text(j, i, value,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    # Axis labels
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel(f'Predicted label\naccuracy={accuracy:0.4f}; misclass={misclass:0.4f}')
    plt.show()
