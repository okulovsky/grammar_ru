from .architecture import *

class SyntaxStatsFeaturizer(SimpleFeaturizer):
    def __init__(self):
        super(SyntaxStatsFeaturizer, self).__init__('syntax_stats', False)

    def _featurize_inner(self, db: DataBundle):
        sdf = db.src.set_index('word_id')[['sentence_id']]
        children = (db
                    .syntax_fixes
                    .groupby('syntax_parent_id')
                    .size()
                    .feed(lambda z: z.loc[z.index != -1])
                    .to_frame('children')
                    )
        sdf = sdf.merge(children, left_index=True, right_index=True, how='left')

        descendents = db.syntax_closure.groupby('syntax_parent_id').size().to_frame('descendants')
        sdf = sdf.merge(descendents, left_index=True, right_index=True, how='left')

        up_depth = db.syntax_closure.groupby('word_id').distance.max().to_frame('up_depth')
        sdf = sdf.merge(up_depth, left_index=True, right_index=True, how='left')

        down_depth = db.syntax_closure.groupby('syntax_parent_id').distance.max().to_frame('down_depth')
        sdf = sdf.merge(down_depth, left_index=True, right_index=True, how='left')

        sentence_size = db.src.groupby('sentence_id').size().to_frame('sentence_length')
        sdf = sdf.merge(sentence_size, left_on='sentence_id', right_index=True, how='left')

        sdf.fillna(0, inplace=True)
        sdf['descendants_relative'] = sdf.descendants / sdf.sentence_length

        sdf = sdf.drop('sentence_id', axis=1)
        return sdf