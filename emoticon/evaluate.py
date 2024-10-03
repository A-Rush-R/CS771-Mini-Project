from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt

def plot_confusion_matrix(y_true, y_pred, classes, title=None, cmap=plt.cm.Blues):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=cmap)
    plt.show()

def get_classification_report(y_true, y_pred):
    return classification_report(y_true, y_pred, target_names=['Class 0', 'Class 1'])

def Evaluate(y_true, y_pred) :
    
    report = get_classification_report(y_true, y_pred)
    print(report)
    
    plot_confusion_matrix(y_true, y_pred, classes = ['Class 0', 'Class 1'], title='Confusion Matrix')
    
    print(f'Accuracy is {accuracy_score(y_true,y_pred)}')