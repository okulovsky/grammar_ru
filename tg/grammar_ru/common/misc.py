import pandas as pd
from yo_fluq_ds import fluq

def sample_table(df, category_column, example_column, example_amount=10):
    odf = df
    df = df.groupby([category_column, example_column]).size().to_frame('cnt').reset_index()
    df = df.feed(fluq.add_ordering_column(category_column, 'cnt'))
    df = df.loc[df.order < example_amount]
    df = df.pivot_table(index=category_column, columns='order', values=example_column, aggfunc=lambda z: z)
    df = df.fillna('')

    tdf = odf.groupby(category_column).size().to_frame('popularity')
    df = tdf.merge(df, left_index=True, right_index=True)
    df = df.sort_values('popularity', ascending=False)
    return df