import datasets
import pandas as pd

class Empathy(datasets.GeneratorBasedBuilder):

    BUILDER_CONFIG_CLASS=datasets.BuilderConfig

    def _info(self):

        feature = {
            'message_id':datasets.Value('string'),
            'response_id':datasets.Value('string'),
            'article_id': datasets.Value('int32'),
            'empathy': datasets.Value('float'),
            'distress': datasets.Value('float'),
            'essay':datasets.Value('string')
        }

        return datasets.DatasetInfo(
            description='The WASSA Empathy dataset.',
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
            yield index, {
                'message_id': row['message_id'],
                'response_id': row['response_id'],
                'article_id': row['article_id'],
                'empathy': row['empathy'],
                'distress': row['distress'],
                'essay': row['essay']
            }