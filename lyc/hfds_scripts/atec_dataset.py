from dataclasses import dataclass

import datasets
import os

FEATURES = datasets.Features(
    {
        "texta": datasets.Value("string"),
        "textb": datasets.Value("string"),
        "label": datasets.Value("bool")
    }
)

@dataclass
class ATECConfig(datasets.BuilderConfig):
    """BuilderConfig for text files."""

    data_path: str = None
    max_length: int = 512


class ATEC(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIG_CLASS = ATECConfig

    def _info(self):
        return datasets.DatasetInfo(features=FEATURES)

    def _split_generators(self, dl_manager):

        if not self.config.data_path:
            raise ValueError(f"Data path must be specified, but got data_path={self.config.data_path}")

        return [datasets.SplitGenerator(name='atec', gen_kwargs={"data_path": self.config.data_path})]

    def _generate_examples(self, data_path):
        with open(data_path) as f:
            for index, i in enumerate([i for i in f.read().split('\n') if i]):
                _, texta, textb, label = i.split('\t')
                yield index, {
                    'texta': texta,
                    'textb': textb,
                    'label': int(label)
                }

