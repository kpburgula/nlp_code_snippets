from textacy.ke import sgrank, textrank
from textacy.ke.utils import aggregate_term_variants
import textacy.ke

def extract_kp(text):
    # spacy model for processing
    nlp = textacy.load_spacy_lang("en_core_web_sm")
    doc = textacy.make_spacy_doc(text, nlp)

    kp_sgrank = set([term for term,weight in sgrank(doc)])
    kp_sgrank = aggregate_term_variants(kp_sgrank)

    kp_textrank = set([term for term,weight in textrank(doc)])
    kp_textrank = aggregate_term_variants(kp_textrank)

    return [kp_textrank, kp_sgrank]


text = open('sample').read()
print(extract_kp(text))