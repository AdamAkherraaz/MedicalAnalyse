import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def create_catplot(df):
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])
    df_cat['total'] = 1
    df_cat = df_cat.groupby(['cardio', "variable", "value"], as_index=False).count()
    
    fig = sns.catplot(x="variable", y="total", data=df_cat, hue="value", kind='bar', col="cardio").fig
    fig.savefig('catplot.png')
    return fig

def create_heatmap(df):
    df_heat = df[
    (df['ap_lo'] <= df['ap_hi']) &
    (df['height'] >= df['height'].quantile(0.025)) &
    (df['height'] <= df['height'].quantile(0.975)) &
    (df['weight'] >= df['weight'].quantile(0.025)) &
    (df['weight'] <= df['weight'].quantile(0.975))]

    corr = df_heat.corr(method="pearson")
    mask = np.triu(corr)
    
    fig, ax = plt.subplots(figsize=(12,12))
    sns.heatmap(corr, linewidths=1, annot=True, mask=mask, fmt='.1f', center=0.08, cbar_kws={"shrink":0.05})
    
    fig.savefig('heatmap.png')
    return fig

if __name__ == '__main__':
    df = pd.read_csv('medical_examination.csv')
    df['overweight'] = df['weight'] / ((df['height'] / 100) **2)
    df['overweight'] = np.where(df['overweight'] < 25, 0, 1)

    df['gluc'] = np.where(df['gluc'] == 1, 0, 1)
    df['cholesterol'] = np.where(df['cholesterol'] == 1, 0, 1)
    
    create_catplot(df)
    create_heatmap(df)
