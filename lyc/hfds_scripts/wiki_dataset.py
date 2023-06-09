from dataclasses import dataclass
import datasets
import pyarrow as pa
import os

FEATURES = datasets.Features(
    {
        'text': datasets.Value('string')
    }
)

@dataclass
class WikiConfig(datasets.BuilderConfig):

    data_path : str = None
    min_sent_length : int = 10
    chunksize : int = 10 << 20
    encoding = 'utf-8'

class wiki(datasets.ArrowBasedBuilder):

    BUILDER_CONFIG_CLASS = WikiConfig
    VERSION = "1.0.0"

    def _info(self):
        return datasets.DatasetInfo(features=FEATURES)

    def _split_generators(self, dl_manager):

        if not self.config.data_path or not os.path.isdir(self.config.data_path):
            raise ValueError(f"Data Dir must be specified, but got data_path={self.config.data_path}")

        return [datasets.SplitGenerator(name='train', gen_kwargs={"data_path": self.config.data_path})]
    
    def _generate_tables(self, data_path):
        batch_idx = 0
        files = os.listdir(data_path)
        files = [os.path.join(data_path, file) for file in files]
        print('Files to process: ', files)
        for file in files:
            with open(file, "r", encoding=self.config.encoding) as f:
                while True:
                    batch = f.read(self.config.chunksize)
                    if not batch:
                        break
                    batch += f.readline()  # finish current line
                    batch = batch.splitlines()
                    batch = [i for i in batch if len(i)>self.config.min_sent_length]
                    pa_table = pa.Table.from_arrays([pa.array(batch)], schema=pa.schema({"text": pa.string()}))
                    yield batch_idx, pa_table
                    batch_idx += 1