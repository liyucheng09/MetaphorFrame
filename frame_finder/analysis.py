import pandas as pd
import numpy as np

class ContextPred:
    def __init__(self, conll_str):
        tokens = np.array([np.array(t.split('\t')) for t in conll_str.split('\n')])
        self.tokens = tokens[:, 0]
        self.labels = tokens[:, 2]
        self.frames = tokens[:, 3]
        self.id = ''.join(self.tokens)
    
    def compare(self, literal_ins):
        not_metaphor_match = []
        is_metaphor_match = []

        for label, context_f, literal_f in zip(self.labels, self.frames, literal_ins.frames):
            if label == '_':
                continue
            elif label == '0':
                not_metaphor_match.append(1 if context_f in literal_f else 0)
            elif label == '1':
                is_metaphor_match.append(1 if context_f in literal_f else 0)
        
        return not_metaphor_match, is_metaphor_match
    
    def to_conll_str(self, extra_frames=None):
        conll_str = ''
        for index, (token, label, frame) in enumerate(zip(self.tokens, self.labels, self.frames)):
            line = f"{token}\t{label}\t{frame}"
            if extra_frames is not None:
                line += '\t'
                line += '\t'.join(extra_frames[index])
            line += '\n'
            conll_str+=line
        conll_str+='\n'
        return conll_str

class LiteralPred:
    def __init__(self, conll_str):
        tokens = [t.split('\t') for t in conll_str.split('\n')]
        self.tokens = []
        self.labels = []
        self.targets = []
        self.frames = []

        for token in tokens:
            self.tokens.append(token[0])
            if token[3] == '0':
                label = '_'
            elif token[3] == '1' and token[4]=='0':
                label = '0'
            else:
                label = '1'
            self.labels.append(label)
            self.targets.append(token[5])
            self.frames.append(token[5:])
        
        self.id = ''.join(self.tokens)


if __name__ == '__main__':

    context_path = 'data/prediction_output.tsv'
    literal_path = 'data/predicted-targets.tsv'

    context_ins = {}
    literal_ins = {}

    with open(context_path, encoding='utf-8') as f:
        for conll_str in f.read().split('\n\n'):
            if not conll_str:
                continue
            ins = ContextPred(conll_str)
            context_ins[ins.id] = ins

    with open(literal_path, encoding='utf-8') as f:
        for conll_str in f.read().split('\n\n'):
            if not conll_str:
                continue
            ins = LiteralPred(conll_str)
            literal_ins[ins.id] = ins
    
    all_not_metaphor = []
    all_is_metaphor = []
    fo = open('pred_out.tsv', 'w', encoding='utf-8')
    for id, ins in context_ins.items():
        l_ins = literal_ins[ins.id]
        conll_str = ins.to_conll_str(l_ins.frames)
        fo.write(conll_str)
        not_m, is_m = ins.compare(l_ins)
        if len(not_m) !=0:
            all_not_metaphor.append(sum(not_m)/len(not_m))
        if len(is_m) !=0:
            all_is_metaphor.append(sum(is_m)/len(is_m))
        # all_not_metaphor.extend(not_m)
        # all_is_metaphor.extend(is_m)
    
    fo.close()
    print(f"is metaphor match:{sum(all_is_metaphor)/len(all_is_metaphor)}, not metaphor match: {sum(all_not_metaphor)/len(all_not_metaphor)}")




