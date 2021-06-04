from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction

# Without smoothing
predicted_sentences = [['He', 'is', 'watching', 'the', 'River', 'Flow'],['looking', 'the', 'River']]
actual_sentence = ['Watching', 'the', 'River', 'Flow']

print(sentence_bleu(predicted_sentences,actual_sentence))

# With smoothing
chencherry = SmoothingFunction().method2
print(sentence_bleu(predicted_sentences,actual_sentence, smoothing_function= chencherry))
