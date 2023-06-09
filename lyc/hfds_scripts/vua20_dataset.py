import datasets
import pandas as pd
import os

class Vua20(datasets.GeneratorBasedBuilder):

    BUILDER_CONFIG_CLASS=datasets.BuilderConfig

    def _info(self):

        if self.config.name == 'default':
            feature = {
                'id':datasets.Value('int32'),
                'sent_id':datasets.Value('string'),
                'word_index': datasets.Value('int32'),
                'tokens':datasets.Sequence(datasets.Value('string')),
                'label':datasets.ClassLabel(num_classes=2, names=['Not Metaphor', 'Is Metaphor']),
                'pos':datasets.Value('string')
            }
        elif self.config.name == 'combined':
            feature = {
                'sent_id':datasets.Value('string'),
                'is_target': datasets.Sequence(datasets.ClassLabel(num_classes=2)),
                'tokens':datasets.Sequence(datasets.Value('string')),
                'labels': datasets.Sequence(datasets.ClassLabel(num_classes=2)),
                'pos_tags':datasets.Sequence(datasets.Value('string'))
            }

        return datasets.DatasetInfo(
            description='Vua metaphor detection datasets.',
            features=datasets.Features(feature),
            config_name=self.config.name
        )
    
    def _split_generators(self, dl_manager):
        # print(self.config)
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={'filepath': os.path.join(self.config.data_dir, 'train.tsv') }),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={'filepath': os.path.join(self.config.data_dir, 'test.tsv')}),
        ]
    
    def _generate_examples(self, filepath):
        df = pd.read_csv(filepath, sep='\t')

        if self.config.name == 'default':
            for index, row in df.iterrows():
                yield index, {
                    'id': index,
                    'sent_id': row['index'],
                    'tokens': row['sentence'].split(),
                    'word_index': row['w_index'],
                    'label': row['label'],
                    'pos': row['POS']
                }
        
        elif self.config.name == 'combined':
            for index, (sent_id, group) in enumerate(df.groupby('index')):
                tokens = group.iloc[0]['sentence'].split()
                is_target = [1 if i in group.w_index.values else 0 for i in range(len(tokens))]
                labels = [1 if is_target[i] and group.label.iloc[sum(is_target[:i+1])-1] else 0 for i in range(len(tokens))]
                pos_tags = [group.POS.iloc[sum(is_target[:i+1])-1] if is_target[i] else '_' for i in range(len(tokens))]

                yield index, {
                    'sent_id': sent_id,
                    'tokens': tokens,
                    'is_target': is_target,
                    'labels': labels,
                    'pos_tags': pos_tags
                }