from yo_fluq_ds import fluq

def make_samples(odf, word_column, group_by_column, n_samples):
    if not isinstance(group_by_column, list):
        group_by_column=[group_by_column]
    df = odf.copy()
    df[group_by_column] = df[group_by_column].fillna('#NONE')
    df = df.groupby([word_column]+group_by_column).size().to_frame('pop').reset_index()
    df = df.feed(fluq.add_ordering_column(group_by_column, ('pop', False)))
    df = df.loc[df.order<n_samples].pivot_table(index=group_by_column, columns='order', values=word_column, aggfunc=lambda z:z)
    df = odf.groupby(group_by_column).size().to_frame('pop').merge(df, left_index=True, right_index=True)
    df = df.sort_values('pop',ascending=False)
    return df