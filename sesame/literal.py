import pickle
import nltk
import pandas as pd
import sys
from nltk.corpus import wordnet

def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''

def get_fn_pos_by_rules(pos, token):
    """
    Rules for mapping NLTK part of speech tags into FrameNet tags, based on co-occurrence
    statistics, since there is not a one-to-one mapping.
    """
    if pos[0] == "v" or pos in ["rp", "ex", "md"]:  # Verbs
        rule_pos = "v"
    elif pos[0] == "n" or pos in ["$", ":", "sym", "uh", "wp"]:  # Nouns
        rule_pos = "n"
    elif pos[0] == "j" or pos in ["ls", "pdt", "rbr", "rbs", "prp"]:  # Adjectives
        rule_pos = "a"
    elif pos == "cc":  # Conjunctions
        rule_pos = "c"
    elif pos in ["to", "in"]:  # Prepositions
        rule_pos = "prep"
    elif pos in ["dt", "wdt"]:  # Determinors
        rule_pos = "art"
    elif pos in ["rb", "wrb"]:  # Adverbs
        rule_pos = "adv"
    elif pos == "cd":  # Cardinal Numbers
        rule_pos = "num"
    else:
        sys.stderr.write("WARNING: Rule not defined for part-of-speech {} word {} - treating as noun.\n".format(pos, token))
        return "n"
    return rule_pos

if __name__ == '__main__':
    tsv_file = '/Users/liyucheng/latex/My_paper/acl22/VUA-Frames.tsv'
    out_file = '/Users/liyucheng/latex/My_paper/acl22/VUA-literal-frame-1.tsv'
    out_file = open(out_file, 'w', encoding='utf-8')
    with open(tsv_file, 'r', encoding='utf-8') as f:
        data = f.read().split('\n\t\t\t\t\t\t\t\t\n')
    
    with open('sesame/lu2frame.pkl', 'rb') as f:
        lu2frame = pickle.load(f)
    
    lemmatizer = nltk.stem.WordNetLemmatizer()

    for sent in data:
        if not sent:
            continue
        tokens = [i.split('\t') for i in sent.split('\n')]
        words = [t[0].lower() for t in tokens]
        is_targets = [t[1] for t in tokens]
        
        pos_tags = [p[1] for p in nltk.pos_tag(words)]
        lemmas = []
        for word, tag in nltk.pos_tag(words):
            wntag = get_wordnet_pos(tag)
            if not wntag:
                lemmas.append(lemmatizer.lemmatize(word))
                continue
            lemmas.append(lemmatizer.lemmatize(word, wntag))
        # lemmas = [ lemmatizer.lemmatize(w, pos) for w,pos in zip(words, pos_tags)]
        pos_tags = [ get_fn_pos_by_rules(pos.lower(), token) for pos, token in zip(pos_tags, words)]

        potential_targets = [ lemma+'.'+pos  for lemma, pos in zip(lemmas, pos_tags)]

        for is_target, target, word in zip(is_targets, potential_targets, words):
            if is_target == '1':
                if target in lu2frame:
                    literal_frame = '\t'.join(lu2frame[target])
                else:
                    literal_frame = '_'
            else:
                literal_frame=''
            out_file.write(f'{word}\t{target}\t{literal_frame}\n')
        out_file.write('\n')
    out_file.close()

