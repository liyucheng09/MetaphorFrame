import datasets
import pandas as pd

class Hard(datasets.GeneratorBasedBuilder):

    BUILDER_CONFIG_CLASS=datasets.BuilderConfig

    def _info(self):

        feature = {
            'id':datasets.Value('int32'),
            'word_index': datasets.Value('int32'),
            'lemma': datasets.Value('string'),
            'word': datasets.Value('string'),
            'tokens':datasets.Sequence(datasets.Value('string')),
            'label':datasets.ClassLabel(num_classes=2, names=['literal', 'metaphorical'])
        }

        return datasets.DatasetInfo(
            description='The Hard Metaphor dataset.',
            features=datasets.Features(feature),
        )
    
    def _split_generators(self, dl_manager):

        return datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={'filepath': self.config.data_files}),
        # return [
        #     datasets.SplitGenerator(name='hard', gen_kwargs={'filepath': self.config.data_path}),
        # ]
    
    def _generate_examples(self, filepath):
        df = pd.read_csv(filepath, sep='\t')

        for index, row in df.iterrows():
            tokens = [token if not (token.startswith('[') and token.endswith(']')) else token[1:-1] for token in row['context'].split()]
            yield index, {
                'id': index,
                'word': row['word'],
                'lemma': row['lemma'] if not row['lemma'].startswith('*') else row['lemma'].strip('*'),
                'tokens': tokens,
                'word_index': row['target_index'],
                'label': row['label']
            }