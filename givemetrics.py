from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import ipywidgets
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import pandas as pd

# This is a function which prints selected metrics for a previously trained model.
def plot_roc(y_obs, y_proba, ax = None):
    fpr, tpr, _ = metrics.roc_curve(y_obs, y_proba)
    fig = plt.figure()
    if ax is None:
      ax = plt.gca()
    ax.plot(fpr, tpr, color='darkorange', lw=2, label='AUC = %0.2f)' % metrics.roc_auc_score(y_obs, y_proba))
    ax.plot([0, 1], [0, 1], color='navy', lw=.7, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic')
    ax.legend(loc="lower right")
    fig.show()
    return fig

def plot_pre_rec(y_obs, y_proba, ax = None):
    precision, recall, _ = metrics.precision_recall_curve(y_obs, y_proba)
    fig = plt.figure()
    if ax is None:
      ax = plt.gca()
    ax.step(recall, precision, color = 'b', alpha = 0.2, where = 'post', label='AP=%0.2f' % metrics.average_precision_score(y_obs, y_proba))
    ax.fill_between(recall, precision, color = 'b', step = 'post', alpha = 0.2)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_title('Presicion-Recall curve')
    ax.legend(loc ='lower left')
    fig.show()
    return fig

def metric_summary(y_obs, y_pred, y_proba, kind = ''):
    return """
{} evaluation of Classifier
Confusion matrix:
{}
Accuracy: {:.4f}
Precision: {:.4f}
Recall: {:.4f}
F1: {:.4f}
Matthew's Correlation: {:.4f}
ROC AUC: {:.4f}""".format(kind, 
                          metrics.confusion_matrix(y_obs, y_pred),
                          metrics.accuracy_score(y_obs, y_pred),
                          metrics.precision_score(y_obs, y_pred),
                          metrics.recall_score(y_obs, y_pred),
                          metrics.f1_score(y_obs, y_pred),
                          metrics.matthews_corrcoef(y_obs, y_pred),
                          metrics.roc_auc_score(y_obs, y_proba))

def give_metrics(test_y_obs, test_y_pred, test_y_proba, train_y_obs, train_y_pred, train_y_proba):
    print (metric_summary(test_y_obs, test_y_pred, test_y_proba, kind='TEST'))
    print(metric_summary(train_y_obs, train_y_pred, train_y_proba, kind='TRAINING'))
    fig, (ax2, ax3) = plt.subplots(2,2,figsize=(8,5))
    ax2[0].text(0,1.2,metric_summary(test_y_obs, test_y_pred, test_y_proba, kind='TEST'))
    plot_roc(test_y_obs, test_y_proba, ax=ax2[0])
    plot_pre_rec(test_y_obs, test_y_proba, ax=ax3[0])
    ax2[1].text(0,1.2,metric_summary(train_y_obs, train_y_pred, train_y_proba, kind='TRAINING'))
    plot_roc(train_y_obs, train_y_proba, ax=ax2[1])
    plot_pre_rec(train_y_obs, train_y_proba, ax3[1])
    fig.tight_layout()
    return fig
    
def precisionplots(predproblist, ytruelist):
  fig, ax1 = plt.subplots()
  ax1.set_ylabel('Präzision', color = 'b')
  ax1.set_xlabel('Schwellenwert für Klassifikation')
  ax1.plot(np.arange(0, 1, 0.01),
            [metrics.precision_score(ytruelist,
                                     [1 if p >= th else 0 for p in predproblist])
            for th in np.arange(0, 1, 0.01)],
            c = 'b')

  rects = ax1.bar(np.arange(0, 1.1, 0.1),
           [metrics.precision_score(ytruelist,
                                    [1 if p >= th else 0 for p in predproblist])
            for th in np.arange(0, 1.1, 0.1)],
            0.09,
            alpha = 0.5)
  th = 0
  for rect in rects:
    height = rect.get_height()
    ax1.text(rect.get_x()+rect.get_width()/2.0, height + 0.01,
             '%d' % int(sum([p >= th for p in predproblist])), 
             ha = 'center', va = 'bottom')
    th += .1
  ax2 = ax1.twinx()
  ax2.set_ylabel('Mengen', color = 'r')
  ax2.plot(np.arange(0, 1, 0.01),
           [sum([p >= th for p in predproblist]) 
            for th in np.arange(0, 1, 0.01)],
            c = 'r')
  plt.title('Präzision der Vorhersage')
  fig.tight_layout()
  plt.show()
  return fig

def visResDrops(y_pred, y_true):
    plt.scatter(np.arange(len(y_true))[y_true==0], y_pred[y_true==0], edgecolor='black', s=15, label="0", alpha=0.2)
    plt.scatter(np.arange(len(y_true))[y_true==1], y_pred[y_true==1], edgecolor='black', s=15, label="1", color="yellow", alpha=0.2)

def visResThresh(threshold, y_pred, y_true):
    visResDrops(y_pred, y_true)
    plt.hlines(threshold, plt.gca().get_xlim()[0], plt.gca().get_xlim()[1], colors="r", zorder=100, label="Schwellwert", linestyles="-")
    #plt.legend()
    y_pred_ = y_pred.copy()
    y_pred_[y_pred_ < threshold] = 0
    y_pred_[y_pred_ >= threshold] = 1
    print(confusion_matrix(y_true, y_pred_))
    print("precision score: %f" % precision_score(y_true, y_pred_))
    print("recall score: %f" % recall_score(y_true, y_pred_))
    plt.show()
    
def ShowVisRes(y_pred, y_test):
  ipywidgets.interact(lambda threshold: visResThresh(threshold , y_pred[:], y_test[:]), 
                    threshold=ipywidgets.FloatSlider(min=0,max=1,step=0.01,value=0.5))

def pred_hist(model, testset, columns=[], cat_cols=[], threshold = 0.5):
  #gives a histogram of features in columns, 
  #selected on those instances with a predition proba higher than threshold
  df = pd.DataFrame(testset)
  df['prediction'] = model.predict_proba(testset)[:,1]
  for c in columns:
    df[df.prediction >= threshold][c].plot.hist()
    plt.title('Histogramm '+c+' erwarteter Abschlüsse mit Wahrscheinlichkeit '+str(threshold))
    plt.show()
  for list in cat_cols:
    values = []
    for c in list:
      values.append(df[df.prediction >= threshold][c].sum())
    plt.pie(values, labels=list, autopct='%1.1f%%')
    plt.title('Erwartete Abschlüsse bei Wahrscheinlichkeit '+str(threshold))
    plt.show()
    
def compare_preds(model1, model2, testset1, on1, testset2=[], on2=None):
    #like a Venn-diagram: visualizes how many instances are predicted by each or both or none
    #of two classifiers against the probability threshold
    if not on2:
      on2 = on1
    if not any(testset2):
      testset2=testset1
    df1 = pd.DataFrame({on1: testset1[on1]})
    df2 = pd.DataFrame({on2: testset2[on2]})
    df1['model1_proba'] = model1.predict_proba(testset1.drop(on1, axis=1))[:,1]
    df2['model2_proba'] = model2.predict_proba(testset2.drop(on2, axis=1))[:,1]
    df=df1.merge(df2, left_on=on1, right_on=on2, how='inner')
    labels = ['both', 'model1', 'model2', 'none']
    thresholds = np.arange(0,1,0.1)
    fig, ax = plt.subplots()
    ax.stackplot(thresholds,
                 [[df.loc[(df.model1_proba>=th) & (df.model2_proba>=th)].count()[0] for th in thresholds],
                  [df.loc[(df.model1_proba>=th) & (df.model2_proba<th)].count()[0] for th in thresholds],
                  [df.loc[(df.model1_proba<th) & (df.model2_proba>=th)].count()[0] for th in thresholds],
                  [df.loc[(df.model1_proba<th) & (df.model2_proba<th)].count()[0] for th in thresholds]],
                 labels=labels)
    plt.xlabel('Wahrscheinlichkeit')
    plt.ylabel('Anzahl')
    plt.title('Vergleich der  Vorhersagen')
    ax.legend(loc=0)
    plt.show()

    
def featchart(feats, scores, names):
  #bar chart of feature importances
  #scores are normalized for visual comparability
  ind = np.arange(len(feats))  # the x locations for the groups
  width = 0.2  # the width of the bars
  scores = [[float(i)/max(raw) for i in raw] for raw in scores]
  fig, ax = plt.subplots()
    
  for i in range(len(scores)):
    rects = ax.barh(ind - (2*i/(len(scores)-1) -1)*1.5*width, scores[i], width, label=names[i])
  
  #box = ax.get_position()
  #ax.set_position([box.x0, box.y0, box.width * 0.6, box.height])
  ax.set_xlabel('Normalized Scores')
  ax.set_ylabel('Features')
  ax.set_title('Feature Importances')
  ax.invert_yaxis()  
  ax.set_yticks(ind)
  ax.set_yticklabels(feats)
  ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
  #plt.subplots_adjust(right=0.7)
  plt.tight_layout(pad=0)
  plt.show()
  return fig