# -*- coding: utf-8 -*-
import pandas as pd
import spacy
import codecs
import os
import glob
import argparse
def get_ARC_test_data(ARC_Easy_path, ARC_Challenge_path, plain_text_path):
    pd_easy = pd.read_csv(ARC_Easy_path)
    pd_challenge = pd.read_csv(ARC_Challenge_path)
    for _, row in pd_easy.iterrows():
        question_answer = row['question']
        if '(3)' in question_answer and '(4)' not in question_answer:
            question = question_answer.split('(1)')[0].strip()
            answerA = question_answer.split('(1)')[1].split('(2)')[0].strip()
            answerB = question_answer.split('(2)')[1].split('(3)')[0].strip()
            answerC = question_answer.split('(3)')[1].strip()
            answerD = None
            answerE = None
        elif '(1)' and '(2)' and '(3)' in question_answer:
            question = question_answer.split('(1)')[0].strip()
            answerA = question_answer.split('(1)')[1].split('(2)')[0].strip()
            answerB = question_answer.split('(2)')[1].split('(3)')[0].strip()
            answerC = question_answer.split('(3)')[1].split('(4)')[0].strip()
            answerD = question_answer.split('(4)')[1].strip()
            answerE = None
        elif '(D)' not in question_answer:
            question = question_answer.split('(A)')[0].strip()
            answerA = question_answer.split('(A)')[1].split('(B)')[0].strip()
            answerB = question_answer.split('(B)')[1].split('(C)')[0].strip()
            answerC = question_answer.split('(C)')[1].strip()
            answerD = None
            answerE = None
        elif '(E)' not in question_answer:
            question = question_answer.split('(A)')[0].strip()
            answerA = question_answer.split('(A)')[1].split('(B)')[0].strip()
            answerB = question_answer.split('(B)')[1].split('(C)')[0].strip()
            answerC = question_answer.split('(C)')[1].split('(D)')[0].strip()
            answerD = question_answer.split('(D)')[1].strip()
            answerE = None
        else:
            question = question_answer.split('(A)')[0].strip()
            answerA = question_answer.split('(A)')[1].split('(B)')[0].strip()
            answerB = question_answer.split('(B)')[1].split('(C)')[0].strip()
            answerC = question_answer.split('(C)')[1].split('(D)')[0].strip()
            answerD = question_answer.split('(D)')[1].split('(E)')[0].strip()
            answerE = question_answer.split('(E)')[1].strip()
        with open(os.path.join(plain_text_path, row['questionID'])+'.txt','w+') as file:
            file.write(question + '\n')
            file.write(answerA + '\n')
            file.write(answerB + '\n')
            file.write(answerC + '\n')
            if answerD is not None:
                file.write(answerD + '\n')
            if answerE is not None:
                file.write(answerE + '\n')
    for _, row in pd_challenge.iterrows():
        question_answer = row['question']
        if '(3)' in question_answer and '(4)' not in question_answer:
            question = question_answer.split('(1)')[0].strip()
            answerA = question_answer.split('(1)')[1].split('(2)')[0].strip()
            answerB = question_answer.split('(2)')[1].split('(3)')[0].strip()
            answerC = question_answer.split('(3)')[1].strip()
            answerD = None
            answerE = None
        elif '(1)' and '(2)' and '(3)' in question_answer:
            question = question_answer.split('(1)')[0].strip()
            answerA = question_answer.split('(1)')[1].split('(2)')[0].strip()
            answerB = question_answer.split('(2)')[1].split('(3)')[0].strip()
            answerC = question_answer.split('(3)')[1].split('(4)')[0].strip()
            answerD = question_answer.split('(4)')[1].strip()
            answerE = None
        elif '(D)' not in question_answer:
            question = question_answer.split('(A)')[0].strip()
            answerA = question_answer.split('(A)')[1].split('(B)')[0].strip()
            answerB = question_answer.split('(B)')[1].split('(C)')[0].strip()
            answerC = question_answer.split('(C)')[1].strip()
            answerD = None
            answerE = None
        elif '(E)' not in question_answer:
            question = question_answer.split('(A)')[0].strip()
            answerA = question_answer.split('(A)')[1].split('(B)')[0].strip()
            answerB = question_answer.split('(B)')[1].split('(C)')[0].strip()
            answerC = question_answer.split('(C)')[1].split('(D)')[0].strip()
            answerD = question_answer.split('(D)')[1].strip()
            answerE = None
        else:
            question = question_answer.split('(A)')[0].strip()
            answerA = question_answer.split('(A)')[1].split('(B)')[0].strip()
            answerB = question_answer.split('(B)')[1].split('(C)')[0].strip()
            answerC = question_answer.split('(C)')[1].split('(D)')[0].strip()
            answerD = question_answer.split('(D)')[1].split('(E)')[0].strip()
            answerE = question_answer.split('(E)')[1].strip()

        with open(os.path.join(plain_text_path, row['questionID'])+'.txt', 'w+') as file:
            file.write(question + '\n')
            file.write(answerA + '\n')
            file.write(answerB + '\n')
            file.write(answerC + '\n')
            if answerD is not None:
                file.write(answerD + '\n')
            if answerE is not None:
                file.write(answerE + '\n')
            #file.write('\n')

def get_start_and_end_offset_of_token_from_spacy(temp, token):
    start = temp + token.idx
    end = start + len(token)
    return start, end

def get_sentences_and_tokens_from_spacy(plain_text_path, spacy_nlp):
    with open(plain_text_path, 'r') as f:
        sentences = []
        for line in f.readlines():
            if len(line.strip())==0 or line[0] == '\n' or line.strip() == '':
                continue
            document = spacy_nlp(line.strip())
            for span in document.sents:
                sentence=[document[i] for i in range(span.start, span.end)]
                sentence_tokens = []
                for token in sentence:
                    token_dict = {}
                    token_dict['pos'] = token.pos_
                    token_dict['text'] = str(token)
                    if token_dict['text'].strip() in ['\n', '\t', ' ', '']:
                        continue
                    sentence_tokens.append(token_dict)
                sentences.append(sentence_tokens)
    return sentences

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ARC_data_dir", default=None, type=str, required=True,
                        help="The ARC data dir downloaded from AI2 website. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--ARCtest_plaintext_dir", default='./ARCtest_plaintext', type=str,
                        help="The ner data dir, it should be brat format and contains many txt files and ann files")
    parser.add_argument("--ARCtest_conll_output_dir", default='./conll_data', type=str,
                        help="The chunked brat format data output dir, it should be brat format and contains many txt files and ann files")
    args = parser.parse_args()
    if not os.path.exists(args.ARCtest_plaintext_dir):
        os.makedirs(args.ARCtest_plaintext_dir)
    get_ARC_test_data(os.path.join(args.ARC_data_dir, 'ARC-Easy/ARC-Easy-Test.csv'),
                      os.path.join(args.ARC_data_dir, 'ARC-Challenge/ARC-Challenge-Test.csv'), args.ARCtest_plaintext_dir)
    spacy_nlp = spacy.load('en_core_web_sm', disable=["parser", "ner", "entity_linker", "textcat", "entity_ruler"])
    spacy_nlp.add_pipe(spacy_nlp.create_pipe('sentencizer'))
    text_filepaths = sorted(glob.glob(os.path.join(args.ARCtest_plaintext_dir, '*.txt')))
    output_file = codecs.open(os.path.join(args.ARCtest_conll_output_dir,'ARC_test_spacy.txt'), 'w', 'utf-8')
    numProcessed = 0
    numSkippedQuestions = 0
    for text_filepath in text_filepaths:
        numProcessed += 1
        base_filename = os.path.splitext(os.path.basename(text_filepath))[0]
        sentences = get_sentences_and_tokens_from_spacy(text_filepath, spacy_nlp)
        if (len(sentences) == 0):
            print("Error loading annotation -- skipping this question. ")
            numSkippedQuestions += 1
            continue
        for sentence in sentences:
            for token in sentence:
                output_file.write(
                    '{0} {1} {2}\n'.format(token['text'], base_filename, token['pos']))
            output_file.write('\n')

    output_file.close()
    print('Done.')
    print("Number of skipped questions: " + str(numSkippedQuestions))
    del spacy_nlp

if __name__ == '__main__':
    main()
