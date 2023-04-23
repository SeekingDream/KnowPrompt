import nltk
import torch
from nltk.tokenize import word_tokenize, sent_tokenize
from supar import Parser
import string
import random
from nltk.tokenize.treebank import TreebankWordDetokenizer
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from scipy.stats import entropy
import json
from .abstract_tuner import AbstractTuner


def detokenize(tokens):
    return TreebankWordDetokenizer().detokenize(tokens)


def check_child(tree):
    check = False
    count = 0
    total_count = 0
    for subtree in tree:
        total_count += 1
        if type(subtree) == nltk.tree.Tree:
            if subtree.label() == '_':
                count += 1
    if count >= total_count - count: check = True

    return check


def collect_leaves(parsed_tree):
    leaves = []
    for tree in parsed_tree:
        if type(parsed_tree) != nltk.tree.Tree: continue
        if tree.label() == '_':
            leaves.append(detokenize(tree.leaves()))
            continue
        if check_child(tree):
            leaves.append(detokenize(tree.leaves()))
        else:
            leaves.extend(collect_leaves(tree))
    return leaves


class GribsTuner(AbstractTuner):

    def get_response(self, input_text, num_return_sequences, num_beams):
        batch = self.para_tokenizer([input_text], truncation=True, padding='longest', max_length=60, return_tensors="pt").to(
            self.device)
        translated = self.para_model.generate(
            **batch, max_length=60, num_beams=num_beams,
            num_return_sequences=num_return_sequences, temperature=1.5
        )
        tgt_text = self.para_tokenizer.batch_decode(translated, skip_special_tokens=True)
        return tgt_text

    def acc_score_func(self, prompt):
        return 0

    def delete_phrase(self, candidate, phrase):
        if candidate.find(' ' + phrase) > 0:
            answer = candidate.replace(' ' + phrase, ' ')
        elif candidate.find(phrase + ' ') > 0:
            answer = candidate.replace(phrase + ' ', ' ')
        else:
            answer = candidate.replace(phrase, '')
        return answer

    def add_phrase(self, candidate, phrase, after):
        if after == '':
            answer = phrase + ' ' + candidate
        else:
            if candidate.find(' ' + after) > 0:
                answer = candidate.replace(' ' + after, ' ' + after + ' ' + phrase)
            elif candidate.find(after + ' ') > 0:
                answer = candidate.replace(after + ' ', after + ' ' + phrase + ' ')
            else:
                answer = candidate.replace(after, after + phrase)
        return answer

    def swap_phrases(self, candidate, phrase_1, phrase_2):
        if candidate.find(' ' + phrase_1 + ' ') >= 0:
            answer = candidate.replace(' ' + phrase_1 + ' ', ' <1> ')
        else:
            answer = candidate.replace(phrase_1, '<1>')
        if candidate.find(' ' + phrase_2 + ' ') >= 0:
            answer = candidate.replace(' ' + phrase_2 + ' ', ' <2> ')
        else:
            answer = candidate.replace(phrase_2, '<2>')
        answer = answer.replace('<1>', phrase_2)
        answer = answer.replace('<2>', phrase_1)
        return answer

    def substitute_phrase(self, candidate, phrase):
        num_beams = 10
        num_return_sequences = 10
        paraphrases = self.get_response(phrase, num_return_sequences, num_beams)
        paraphrase = np.random.choice(paraphrases, 1)[0]
        paraphrase = paraphrase.strip('.')
        if candidate.find(' ' + phrase) > 0:
            answer = candidate.replace(' ' + phrase, ' ' + paraphrase)
        elif candidate.find(phrase + ' ') > 0:
            answer = candidate.replace(phrase + ' ', paraphrase + ' ')
        else:
            answer = candidate.replace(phrase, paraphrase)
        return answer

    def __init__(self, train_data, eval_data, init_prompt, llm_func, config):
        super(GribsTuner, self).__init__(train_data, eval_data, init_prompt, llm_func, config)
        self.device = torch.device('cuda')
        self.parser = Parser.load('crf-con-en')
        self.level = 'phrase'
        self.edit_operations = ['del', 'swap', 'sub', 'add']
        self.num_compose = 1
        self.num_candidates = 5
        self.epochs = 10
        self.patience = 3
        self.score_func = self.acc_score_func

        para_model_name = 'tuner007/pegasus_paraphrase'
        self.para_tokenizer = PegasusTokenizer.from_pretrained(para_model_name)
        self.para_model = PegasusForConditionalGeneration.from_pretrained(para_model_name).to(self.device).eval()

    def collect_leaves(self, parsed_tree):
        leaves = []
        for tree in parsed_tree:
            if type(parsed_tree) != nltk.tree.Tree: continue
            if tree.label() == '_':
                leaves.append(detokenize(tree.leaves()))
                continue
            if check_child(tree):
                leaves.append(detokenize(tree.leaves()))
            else:
                leaves.extend(collect_leaves(tree))
        return leaves

    def get_phrases(self, instruction):  # one possible way of obtaining disjoint phrases
        phrases = []
        for sentence in sent_tokenize(instruction):
            parsed_tree = self.parser.predict(word_tokenize(sentence), verbose=False).sentences[0].trees[0]
            leaves = collect_leaves(parsed_tree)
            phrases.extend(leaves)
        phrases = [detokenize(word_tokenize(phrase)) for phrase in phrases if
                   phrase not in string.punctuation or phrase == '']
        return phrases

    def get_phrase_lookup(self, base_candidate):
        if self.level == 'phrase':
            phrase_lookup = {p: phrase for p, phrase in enumerate(self.get_phrases(base_candidate))}
        elif self.level == 'word':
            words = word_tokenize(base_candidate)
            words = [w for w in words if w not in string.punctuation or w != '']
            phrase_lookup = {p: phrase for p, phrase in enumerate(words)}
        elif self.level == 'sentence':
            sentences = sent_tokenize(base_candidate)
            phrase_lookup = {p: phrase for p, phrase in enumerate(sentences)}
        elif self.level == 'span':
            phrases = []
            for sentence in sent_tokenize(base_candidate):
                spans_per_sentence = np.random.choice(range(2, 5))  # split sentence into 2, 3, 4, 5 chunks
                spans = np.array_split(word_tokenize(sentence), spans_per_sentence)
                spans = [detokenize(s) for s in spans]
                phrases.extend(spans)
            phrase_lookup = {p: phrase for p, phrase in enumerate(phrases)}
        else:
            raise ValueError()
        return phrase_lookup

    def get_edit_op(self):
        if self.num_compose == 1:
            edits = np.random.choice(self.edit_operations, self.num_candidates)
        else:
            edits = []
            for n in range(self.num_candidates):
                edits.append(np.random.choice(self.edit_operations, self.num_compose))
        return edits

    def perform_edit(self, edit, base, phrase_lookup, delete_tracker):
        if edit == 'del':
            [i] = np.random.choice(list(phrase_lookup.keys()), 1)
            return self.delete_phrase(base, phrase_lookup[i]), [i]
        elif edit == 'swap':
            try:
                [i, j] = np.random.choice(list(phrase_lookup.keys()), 2, replace=False)
            except:
                [i, j] = np.random.choice(list(phrase_lookup.keys()), 2, replace=True)
            return self.swap_phrases(base, phrase_lookup[i], phrase_lookup[j]), [i, j]
        elif edit == 'sub':
            [i] = np.random.choice(list(phrase_lookup.keys()), 1)
            return self.substitute_phrase(base, phrase_lookup[i]), [i]
        elif edit == 'add':
            keys = list(phrase_lookup.keys())
            keys.append(-1)
            [i] = np.random.choice(keys, 1)
            if i >= 0:
                after = phrase_lookup[i]
            else:
                after = ''
            if len(delete_tracker) == 0: return base, []
            phrase = np.random.choice(delete_tracker, 1)[0]
            return self.add_phrase(base, phrase, after), [phrase]

    def tune(self):
        operations_tracker = []
        base_candidate = detokenize(word_tokenize(self.init_prompt))
        assert word_tokenize(base_candidate) == word_tokenize(self.init_prompt)
        original_candidate = base_candidate

        base_score = self.score_func(base_candidate)

        delete_tracker = []
        patience_counter = 1
        for i in range(self.epochs):
            deleted = {}
            added = {}
            phrase_lookup = self.get_phrase_lookup(base_candidate)
            # if base_candidate == original_candidate:
            #     for p in phrase_lookup.values():
            #         print(p)

            edits = self.get_edit_op()
            print(edits)

            # generate candidates
            candidates = []
            for edit in edits:
                if isinstance(edit, str):

                    candidate, indices = self.perform_edit(edit, base_candidate, phrase_lookup, delete_tracker)

                    candidates.append(candidate)
                    if edit == 'del':
                        deleted[candidate] = [phrase_lookup[indices[0]]]
                    if edit == 'add':
                        if len(indices):
                            added[candidate] = indices
                else:

                    old_candidate = base_candidate
                    composed_deletes = []
                    composed_adds = []
                    for op in edit:
                        phrase_lookup = self.get_phrase_lookup(old_candidate)
                        new_candidate, indices = self.perform_edit(op, old_candidate, phrase_lookup, delete_tracker)
                        if op == 'del':
                            composed_deletes.append(phrase_lookup[indices[0]])
                        if op == 'add':
                            if len(indices):
                                composed_adds.append(indices[0])
                        old_candidate = new_candidate

                    candidates.append(new_candidate)
                    if 'del' in edit:
                        deleted[new_candidate] = composed_deletes
                    if 'add' in edit and len(composed_adds) > 0:
                        added[new_candidate] = composed_adds
            scores = []
            print('candidates',)
            for c in candidates:
                print(c)

            for c, candidate in enumerate(candidates):
                scores.append(self.score_func(candidate))
                print(scores[-1])

            best_idx = np.argmax(scores)
            best_score = scores[best_idx]
            if best_score > base_score:
                patience_counter = 1
                base_candidate = candidates[best_idx]
                base_score = best_score
                operations_tracker.append(edits[best_idx])

                print('New Base Candidate: ', base_candidate)
                if base_candidate in added.keys():
                    print('Notice! Prev tracker: ', delete_tracker)
                    for chunk in added[base_candidate]:
                        try:
                            delete_tracker.remove(chunk)
                        except:
                            pass
                    print('Notice! New tracker: ', delete_tracker)
                if base_candidate in deleted.keys():
                    delete_tracker.extend(deleted[base_candidate])
                base_candidate = detokenize(word_tokenize(base_candidate))

            else:
                patience_counter += 1

                # if args.simulated_anneal:
                #     K = 5
                #     T = T_max * np.exp(-i / K)
                #     idx = np.argmax(scores)
                #     chosen_score = scores[idx]
                #     prob = np.exp((chosen_score - base_score) / T)
                #     if np.random.binomial(1, prob):
                #
                #
                #         base_candidate = candidates[idx]
                #         base_score = chosen_score
                #
                #         if base_candidate in added.keys():
                #
                #             for chunk in added[base_candidate]:
                #                 try:
                #                     delete_tracker.remove(chunk)
                #                 except:
                #                     pass
                #
                #         if base_candidate in deleted.keys():
                #             delete_tracker.extend(deleted[base_candidate])
                #         base_candidate = detokenize(word_tokenize(base_candidate))
                #     else:
                #         if patience_counter > self.patience:
                #             print('Ran out of patience')
                #             break
                #         else:
                #             continue
                # else:
                if patience_counter > self.patience:
                    break
                else:
                    continue

    def evaluate_prompt(self):
        pass


