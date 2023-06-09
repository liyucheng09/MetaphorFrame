import datasets
import xml.etree.cElementTree as ET
from glob import glob
import os
import gc
# from memory_profiler import profile
import objgraph

_UFSAC_FILE = 'ufsac-public-2.1.tar.xz'

class UFSAC(datasets.GeneratorBasedBuilder):

    BUILDER_CONFIG_CLASS = datasets.BuilderConfig

    def _info(self):
        feature = {
            'tokens': datasets.Sequence(datasets.Value('string')),
            'lemmas': datasets.Sequence(datasets.Value('string')),
            'pos_tags': datasets.Sequence(datasets.Value('string')),
            'target_idx': datasets.Value('int32'),
            'sense_keys': datasets.Sequence(datasets.Value('string')),
        }

        return datasets.DatasetInfo(
            features=datasets.Features(feature),
            description = 'UFSAC: the unified Sense Annotated Corpora and Tool'
        )
    
    def _split_generators(self, dl_manager):
        data_dir = dl_manager.download_and_extract(_UFSAC_FILE)
        return datasets.SplitGenerator(name = datasets.Split.TRAIN, gen_kwargs={'data_dir': data_dir}),
    
    def _generate_examples(self, data_dir):
        used_sents = set()
        count = 0
        for file in glob(os.path.join(data_dir, 'ufsac-public-2.1/*.xml')):
            context = ET.iterparse(file, events=('start', 'end'))
            event, root = next(context)
            for event, element in context:
                if element.tag == 'paragraph':
                    para = element
                if element.tag != 'sentence':
                    continue
                if event == 'end' and element.tag == 'sentence':
                    para.remove(element)
                sent = element
                words = sent.findall('word')
                tokens = [token.attrib['surface_form'] if 'surface_form' in token.attrib else '_' for token in words]
                sent_key = ''.join([token.lower() for token in tokens])
                if sent_key in used_sents:
                    continue
                used_sents.add(sent_key)
                lemmas = [token.attrib['lemma'] if 'lemma' in token.attrib else '_' for token in words]
                pos_tags = [token.attrib['pos'] if 'pos' in token.attrib else '_' for token in words]
                for index, word in enumerate(words):
                    if 'wn30_key' in word.attrib:
                        senses = word.attrib['wn30_key'].split(';')
                        yield count, {
                            'tokens': tokens,
                            'lemmas': lemmas,
                            'pos_tags': pos_tags,
                            'target_idx': index,
                            'sense_keys': senses
                        }
                        count+=1