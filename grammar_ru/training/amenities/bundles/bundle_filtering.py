def filter_bundle_by_words(db, word_ids):
    db.data_frames['src'] = db.data_frames['src'].feed(lambda z: z.loc[z.word_id.isin(word_ids)])
    sdf = db.data_frames['src']
    for key in db.data_frames:
        if key == 'src':
            continue
        df = db.data_frames[key]
        index_name = df.index.name
        if index_name in sdf.columns:
            df = df.loc[df.index.isin(sdf[index_name])]
            db.data_frames[key] = df