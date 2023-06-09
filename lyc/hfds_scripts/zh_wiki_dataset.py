from dataclasses import dataclass
import datasets
import pyarrow as pa

FEATURES = datasets.Features(
    {
        'text': datasets.Value('string')
    }
)

@dataclass
class ZHWikiConfig(datasets.BuilderConfig):

    data_path : str = None
    min_sent_length : int = 10
    chunksize : int = 10 << 20
    encoding = 'utf-8'

class zh_wiki(datasets.ArrowBasedBuilder):

    BUILDER_CONFIG_CLASS = ZHWikiConfig
    VERSION = "1.0.0"

    def _info(self):
        return datasets.DatasetInfo(features=FEATURES)

    def _split_generators(self, dl_manager):

        if not self.config.data_path:
            raise ValueError(f"Data path must be specified, but got data_path={self.config.data_path}")

        return [datasets.SplitGenerator(name='train', gen_kwargs={"data_path": self.config.data_path})]
    
    def _generate_tables(self, data_path):
        with open(data_path, "r", encoding=self.config.encoding) as f:
            batch_idx = 0
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