import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, log_loss
from treeinterpreter import treeinterpreter as ti
#import waterfall_chart as waterfall
import waterfall_new as waterfall_new
from tqdm import tqdm
from pdpbox import pdp
import matplotlib.pyplot

#unfortunately, the pandas api extensions don't work. 
#maybe let it inherit.
#@pd.api.extensions.register_dataframe_accessor('rfint')
#class RFintuition(object):
#  def __init__(self, pandas_object):
#    self._obj = pandas_obj
    
#  @property
def score(x1, x2):
  return mean_squared_error(x1, x2)
  #return log_loss(x1, x2)
  
def feat_imp(m, x, y, small_good = True):
  score_list = {}
  score_list['original'] = score(m.predict(x), y)
  imp = {}
  for i in tqdm(range(len(x.columns))):
    rand_idx = np.random.permutation(len(x))
    new_coli = x.values[rand_idx, i]
    new_x = x.copy()
    new_x[x.columns[i]] = new_coli
    score_list [x.columns[i]] = score(m.predict(new_x), y)
    imp [x.columns[i]] = score_list['original'] - score_list[x.columns[i]]
  if small_good:
    return sorted(imp.items(), key=lambda x: x[1])
  else:
    return sorted(imp.items(), key=lambda x: x[1], reverse = True)
    
def pred_ci(model, x_val, y_val, percentile = 95):
  allTree_preds = np.stack([t.predict(x_val) for t in model.estimators_], axis =0)
  err_down = np.percentile(allTree_preds, (100-percentile)/2.0, axis=0)
  err_up = np.percentile(allTree_preds, percentile/2.0, axis=0)
  ci = err_up - err_down
  yhat = model.predict(x_val)
  df = pd.DataFrame()
  df['down'] = err_down
  df['up'] = err_up
  df['y'] = y_val.values
  df['yhat'] = yhat
  df['deviation'] = (df.up - df.down)/df.yhat
  df.reset_index(inplace = True)
  df_sorted = df.iloc[np.argsort(df.deviation)[::-1]]
  return df_sorted


def pred_path_old(model, feat_names, instances, cl=1, plotthr = .04, out = False, viz = True, filename=None):
  feats = []
  prediction, bias, contributions = ti.predict(model, instances)
  for i in range(len(instances)):
    feats.append(sorted(zip(contributions[i,:,cl], feat_names),
                       key=lambda x: -abs(x[0])))
  if out:
    for i in range(len(instances)):
      print("Instance", i)
      print("Bias (trainset mean)", bias[i])
      print("Feature contributions:")
      for c, feature in feats:
          print(feature, round(c, 2))
      print("-"*20)
  if viz:
    plots = []
    for i in range(len(instances)):
      plot = waterfall_new.plot(['bias'] + [f[1] for f in feats[i]],#[:10],
                            [bias[i][1]] + [f[0] for f in feats[i]],#[:10],
                            threshold = plotthr, formatting='{:,.2f}',
                            other_label = 'Andere', sorted_value = True,
                            Title = 'Kunde '+str(i)+': Abschlusswahrscheinlichkeit. %.3f' % prediction[i][1],
                            y_lab = 'Beitrag zur Wahrscheinlichkeit',
                            x_lab = 'Eigenschaften',
                            net_label = 'Vorhersage')
      #waterfall_fig = plot.figure()
      if filename:
        plot.savefig(filename)
    plots.append(plot)#waterfall_fig)
    #waterfall_fig.show()
  return plot

def pred_path(model, feat_names, instances, cl=1, plotthr = .04, nfeats = None, must_feats = None, out = False, viz = True, filename=None):
  feats = []
  feats2display = []
  prediction, bias, contributions = ti.predict(model, instances)
  for i in range(len(instances)):
    feats.append(sorted(zip(contributions[i,:,cl], feat_names),
                        key=lambda x: -abs(x[0])))
    feats2display.append([])
    if must_feats:
      for name in must_feats:
        nameindex = [f[1] for f in feats[i]].index(name)
        feats2display[i].append(feats[i].pop(nameindex))
    if nfeats:
      if nfeats > len(feats2display[i]):
        extendfeats, feats[i] = feats[i][:(nfeats-len(feats2display[i]))],feats[i][(nfeats-len(feats2display[i])):]
        feats2display[i].extend(extendfeats)
    else:
      print('no number and no mustfeats given')
      extendfeats, feats[i] = [f for f in feats[i] if abs(f[0])>=plotthr], [f for f in feats[i] if abs(f[0])<plotthr]
      feats2display[i].extend(extendfeats)
    feats2display[i].sort(key=lambda x: -abs(x[0]))
    feats2display[i].append((sum([f[0] for f in feats[i]]),'Andere'))  
    print(feats2display)
  if out:
    for i in range(len(instances)):
      print("Instance", i)
      print("Bias (trainset mean)", bias[i])
      print("Feature contributions:")
      for c, feature in feats:
          print(feature, round(c, 2))
      print("-"*20)
  if viz:
    plots = []
    for i in range(len(instances)):
      plot = waterfall_new.plot(['bias'] + [f[1] for f in feats2display[i]],#[:10],
                            [bias[i]] + [f[0] for f in feats2display[i]],#[:10],
                            #threshold = plotthr, 
                            formatting='{:,.2f}',
                            other_label = 'Andere', sorted_value = False,
                            Title = 'Kunde '+str(i)+': Abschlusswahrscheinlichkeit. %.3f' % prediction[i],#[cl],
                            y_lab = 'Beitrag zur Wahrscheinlichkeit',
                            x_lab = 'Eigenschaften',
                            net_label = 'Vorhersage')
      #waterfall_fig = plot.figure()
      if filename:
        plot.savefig(filename)
    plots.append(plot)#waterfall_fig)
    #waterfall_fig.show()
  return plot

  
def plot_pdp(model, x, feat, clusters=None, feat_name=None, filename=None):
    feat_name = feat_name or feat
    p = pdp.pdp_isolate(model, x, x.columns, feat)
    fig, ax = pdp.pdp_plot(p, feature_name = feat_name, plot_lines=True, 
                        cluster=clusters is not None, n_cluster_centers=clusters)
    if filename:
      pdp.plt.savefig(filename+'.png', dpi=200)
    pdp.plt.show()
    return fig