# -*- coding: utf-8 -*-
import codecs
import glob
import json
import os
import spacy
import argparse

numWarnings = 0
numProcessed = 0

def replace_unicode_whitespaces_with_ascii_whitespace(string):
    return ' '.join(string.split())

def get_start_and_end_offset_of_token_from_spacy(temp, token):
    start = temp + token.idx
    end = start + len(token)
    return start, end

def get_sentences_and_tokens_from_spacy(text, spacy_nlp):
    global numWarnings

    sentences = []
    temp = 0
    warningOccured = False

    for idx, line in enumerate(text.split('\n')[:-1]):
        document = spacy_nlp(line)
        for span in document.sents:
            sentence = [document[i] for i in range(span.start, span.end)]
            sentence_tokens = []
            for token in sentence:
                token_dict = {}
                token_dict['pos'] = token.pos_
                token_dict['start'], token_dict['end'] = get_start_and_end_offset_of_token_from_spacy(temp, token)
                token_dict['text'] = text[token_dict['start']:token_dict['end']]
                if token_dict['text'].strip() in ['\n', '\t', ' ', '']:
                    continue
                # Make sure that the token text does not contain any space
                if len(token_dict['text'].split(' ')) != 1:
                    print(
                        "WARNING: the text of the token contains space character, replaced with hyphen\n\t{0}\n\t{1}".format(
                            token_dict['text'],
                            token_dict['text'].replace(' ', '-')))
                    token_dict['text'] = token_dict['text'].replace(' ', '-')
                    warningOccured = True

                sentence_tokens.append(token_dict)
                if token == sentence[-1]:
                    temp = token_dict['end'] + 1
            temp_flag = temp
            temp = 0
            sentences.append(sentence_tokens)
        temp = temp_flag

    # If a warning occured in this file, then increment the warning count by 1
    if (warningOccured == True):
        numWarnings += 1
        # return empty sentences
        return []

    return sentences


def get_entities_from_brat(text_filepath, annotation_filepath):
    # load text
    with codecs.open(text_filepath, 'r', 'UTF-8') as f:
        text = f.read()
    # parse annotation file
    entities = []
    with codecs.open(annotation_filepath, 'r', 'UTF-8') as f:
        for line in f.read().splitlines():
            anno = line.split()
            id_anno = anno[0]
            # parse entity
            if id_anno[0] == 'T':
                entity = {}
                entity['id'] = id_anno
                entity['type'] = anno[1]
                entity['start'] = int(anno[2])
                entity['end'] = int(anno[3])
                entity['text'] = ' '.join(anno[4:])
                # Check compatibility between brat text and anootation
                if replace_unicode_whitespaces_with_ascii_whitespace(text[entity['start']:entity['end']]) != \
                        replace_unicode_whitespaces_with_ascii_whitespace(entity['text']):
                    print("Warning: brat text and annotation do not match.")
                    print("\ttext: {0}".format(text[entity['start']:entity['end']]))
                    print("\tanno: {0}".format(entity['text']))
                entities.append(entity)
    return text, entities


def brat_to_conll(input_folder, output_filepath, tokenizer, language):
    global numWarnings
    global numProcessed
    numSkippedQuestions = 0

    if tokenizer == 'spacy':
        spacy_nlp = spacy.load(language, disable=["parser", "ner", "entity_linker", "textcat", "entity_ruler"])
        spacy_nlp.add_pipe(spacy_nlp.create_pipe('sentencizer'))

    else:
        raise ValueError("tokenizer should be 'spacy'.")
    dataset_type = os.path.basename(input_folder)
    print("Formatting {0} set from BRAT to CONLL... ".format(dataset_type), end='')
    text_filepaths = sorted(glob.glob(os.path.join(input_folder, '*.txt')))
    output_file = codecs.open(output_filepath, 'w', 'utf-8')

    numFiles = 0
    for text_filepath in text_filepaths:
        numProcessed += 1
        base_filename = os.path.splitext(os.path.basename(text_filepath))[0]
        annotation_filepath = os.path.join(os.path.dirname(text_filepath), base_filename + '.ann')
        # create annotation file if it does not exist
        if not os.path.exists(annotation_filepath):
            codecs.open(annotation_filepath, 'w', 'UTF-8').close()
        text, entities = get_entities_from_brat(text_filepath, annotation_filepath)
        entities = sorted(entities, key=lambda entity: entity["start"])

        if tokenizer == 'spacy':
            print(base_filename)
            sentences = get_sentences_and_tokens_from_spacy(text, spacy_nlp)

            # If the number of sentences returned was zero, there was an error loading the annotation for this question.
            # In this case, we skip over this question.
            if (len(sentences) == 0):
                print ("Error loading annotation -- skipping this question. ")
                numSkippedQuestions += 1
                continue

        for sentence in sentences:

            for token in sentence:
                token['label'] = []
                for entity in entities:
                    if entity['start'] == token['start'] and token['end']<= entity['end']:
                        token['label'].append('B-{0}'.format(entity['type'].replace('-','_')))
                    elif entity['start'] < token['start'] and token['end']<= entity['end']:
                        token['label'].append('I-{0}'.format(entity['type'].replace('-','_')))
                    elif token['end'] < entity['start']:
                        break
                if token['label'] == []:
                    token['label'] = ['O']

                if len(entities) == 0:
                    entity = {'end': 0}
                output_file.write(
                    '{0} {1} {2} {3} {4} {5}\n'.format(token['text'], base_filename, token['pos'], token['start'], token['end'],
                                                   ' '.join(token['label'])))
            output_file.write('\n')

        # Warning output
        numFiles += 1
        if (numFiles % 10 == 0):
            print("numProcessed:" + str(numProcessed) + "\t Warnings: " + str(numWarnings))

    output_file.close()
    print('Done.')
    print ("Number of skipped questions: " + str(numSkippedQuestions))
    if tokenizer == 'spacy':
        del spacy_nlp

def main():


    parser = argparse.ArgumentParser()
    parser.add_argument("--brat_ner_data_dir", default='./brat_data', type=str,
                        help="The brat format ner data dir which have been chunked and divided already")
    parser.add_argument("--conll_ner_data_output_dir", default='./conll_data', type=str,
                        help="The brat format ner data dir which have been chunked and divided already")
    args = parser.parse_args()

    print("Initializing...")
    print("brat_ner_data_dir: " + args.brat_ner_data_dir)

    if not os.path.exists(args.conll_ner_data_output_dir):
        os.makedirs(args.conll_ner_data_output_dir)
    dataset_brat_folders = {}
    tokenizer = 'spacy'
    for dataset_type in ['train_brat', 'valid_brat', 'test_brat']:
        dataset_brat_folders[dataset_type] = os.path.join(args.brat_ner_data_dir,
                                                          dataset_type)
        if os.path.exists(dataset_brat_folders[dataset_type]) \
                and len(glob.glob(os.path.join(dataset_brat_folders[dataset_type], '*.txt'))) > 0:
            dataset_filepath_for_tokenizer = os.path.join(args.conll_ner_data_output_dir,
                                                          '{0}_{1}.txt'.format(dataset_type.split('_')[0], tokenizer))

            brat_to_conll(dataset_brat_folders[dataset_type],
                          dataset_filepath_for_tokenizer, tokenizer, 'en_core_web_sm')

    print("Warnings: " + str(numWarnings))

if __name__ == '__main__':
    main()