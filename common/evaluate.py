from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt

def plot_confusion_matrix(y_true, y_pred, classes, cmap=plt.cm.Blues, title='Confusion matrix', ax=None):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    
    if ax is None:
        disp.plot(cmap=cmap)
        plt.title(title)
    else:
        disp.plot(cmap=cmap, ax=ax)
        ax.set_title(title)

def get_classification_report(y_true, y_pred):
    return classification_report(y_true, y_pred, target_names=['Class 0', 'Class 1'])

def evaluate_predictions(y_true, y_pred, title, ax=None):
    
    print(f'=========={title}==========')
    report = get_classification_report(y_true, y_pred)
    print(report)
    
    plot_confusion_matrix(y_true, y_pred, classes = ['Class 0', 'Class 1'], title=title, ax = ax)
    
    print(f'Accuracy is {accuracy_score(y_true,y_pred)}')
    print(f'=========={title}==========')