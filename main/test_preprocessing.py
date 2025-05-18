import sys
import unittest

sys.path.append("C:/Users/artur/Desktop/TFG/preprocessing")
sys.path.append("C:/Users/artur/Desktop/TFG/rnn")

from rawtext.rawtext import apply_rt
from sent_segmentation.sent_segmentation import apply_segment
from tokenization.tokenization import tokenize_text
from pos_tag_tokens.pos_tag_tokens import tg_tk
from remove_stopwords.remove_stopwords import stwrd
from stemming.stemming import apply_stemm
from lemmatization.lemmatization import apply_lemm
from line_by_line import modelo2_lbl
from one_word_in_one_word_out import modelo1_owi_owo
from two_words_in_one_word_out import modelo3_twi_owo

class TestPreprocessing(unittest.TestCase):

    def test_rawtext(self):
        raw = apply_rt("C:/Users/artur/Desktop/TFG/textos/textoplano.txt")
        self.assertIsInstance(raw, str)
        self.assertGreater(len(raw), 0)

    def test_segment_sentences(self):
        raw = apply_rt("C:/Users/artur/Desktop/TFG/textos/textoplano.txt")
        sentences = apply_segment(raw)
        self.assertIsInstance(sentences, list)
        self.assertGreater(len(sentences), 0)

    def test_tokenize_text(self):
        raw = apply_rt("C:/Users/artur/Desktop/TFG/textos/textoplano.txt")
        tokens = tokenize_text(raw)
        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)

    def test_pos_tagging(self):
        raw = apply_rt("C:/Users/artur/Desktop/TFG/textos/textoplano.txt")
        tags = tg_tk(raw)
        self.assertIsInstance(tags, list)
        self.assertGreater(len(tags), 0)
        self.assertIsInstance(tags[0], tuple)
        self.assertEqual(len(tags[0]), 3)

    def test_stopwords(self):
        raw = apply_rt("C:/Users/artur/Desktop/TFG/textos/textoplano.txt")
        tokens = tokenize_text(raw)
        filtered = stwrd(tokens)
        self.assertIsInstance(filtered, list)
        self.assertTrue(all(isinstance(t, str) for t in filtered))

    def test_stemming(self):
        raw = apply_rt("C:/Users/artur/Desktop/TFG/textos/textoplano.txt")
        tokens = tokenize_text(raw)
        filtered = stwrd(tokens)
        stems = apply_stemm(filtered)
        self.assertIsInstance(stems, list)
        self.assertTrue(all(isinstance(s, str) for s in stems))

    def test_lemmatization(self):
        raw = apply_rt("C:/Users/artur/Desktop/TFG/textos/textoplano.txt")
        tokens = tokenize_text(raw)
        filtered = stwrd(tokens)
        lemas = apply_lemm(filtered)
        self.assertIsInstance(lemas, list)
        self.assertTrue(all(isinstance(l, str) for l in lemas))

class TestRnn(unittest.TestCase):

    text = "the cat sat on the mat"

    def test_modelo1_output_length(self):
        result = modelo1_owi_owo(self.text, epochs=10, n_words=4, seed_text='the')
        self.assertEqual(len(result.split()), 1 + 4) 

    def test_modelo2_output_length(self):
        result = modelo2_lbl(self.text, epochs=10, n_words=4, seed_text='the cat')
        self.assertEqual(len(result.split()), 2 + 4)  

    def test_modelo3_output_length(self):
        result = modelo3_twi_owo(self.text, epochs=10, n_words=4, seed_text='the cat')
        self.assertEqual(len(result.split()), 2 + 4) 

    def test_modelos_outputs_type(self):
        out1 = modelo1_owi_owo(self.text, epochs=5, n_words=2, seed_text='the')
        out2 = modelo2_lbl(self.text, epochs=5, n_words=2, seed_text='the cat')
        out3 = modelo3_twi_owo(self.text, epochs=5, n_words=2, seed_text='the cat')
        for out in [out1, out2, out3]:
            self.assertTrue(isinstance(out, str))
            self.assertGreater(len(out.strip()), 0)

if __name__ == "__main__":
    unittest.main()