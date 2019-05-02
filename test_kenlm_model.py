from reference_lm.kenlm_model import KenlmModel

MODEL_PATH = 'knlm_model.arpa'
SENT_PATH = 'data_oneb/src-train.txt'


def load_sentences(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return [line.strip() for line in lines]


model = KenlmModel(MODEL_PATH)
sentences = load_sentences(SENT_PATH)

ppl = model.get_ppl(sentences)
print('PPL = {}'.format(ppl))

