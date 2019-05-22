def add1(x):
    if x==-1: return x+1
    else: return x

def add1_revert(x):
    if x==0: return x-1
    else: return x

def encode(df, save_normalized=False):
    li_cat = ['poutcome', 'marital']
    li_cont = ['balance', 'pdays', 'previous', 'campaign']
    fname_norm = "normalized.csv"
 
    ## categorial variables
    df_cat = df.loc[:,li_cat]
    df_cat = pd.get_dummies(df_cat)

    ## continuous variables
    df_cont = df.loc[:, li_cont]
    df_cont.pdays = df_cont.pdays.apply(lambda x: add1(x))
    df_norm = pd.concat([df_cont.mean(), df_cont.max(), df_cont.min()],axis=1)
    df_norm.columns=['meanval', 'maxval', 'minval']

    if save_normalized==True:
        df_norm.to_csv("normalized.csv")

    df_cont = (df_cont - df_cont.mean()) / (df_cont.max() - df_cont.min())
    df_all = pd.concat([df_cat, df_cont], axis=1)
    return df_all


def decode(dfin):
    li_cat = ['poutcome', 'marital']
    li_cont = ['balance', 'pdays', 'previous', 'campaign']
    fname_norm = "normalized.csv"

    df_cont = dfin.loc[:, li_cont]
    df_cat =  dfin.drop(li_cont, axis=1)
    df_norm = pd.read_csv(fname_norm, index_col=0)
    
    df_cont_dec = (df_norm.maxval - df_norm.minval)*df_cont + df_norm.meanval
    df_cont_dec = df_cont_dec.apply(lambda x: round(x,1)).astype(int)
    df_cont_dec.pdays = df_cont_dec.pdays.apply(lambda x: add1_revert(x))

    licol = list(set([i.split('_')[0] for i in df_cat.columns]))
    lidf = []

    for sti in licol:
        li = [i for i in df_cat.columns if sti+'_' in i]
        dfi = df_cat.loc[:,li]
        dic = {i:i.replace(sti+'_', '') for i in dfi.columns}
        dfi = dfi.rename(columns=dic)
        dfi = dfi.idxmax(axis=1)
        dfi.name = sti
        lidf.append(dfi)

    df_cat_dec = pd.concat(lidf, axis=1)

    return pd.concat([df_cat_dec,df_cont_dec], axis=1)
