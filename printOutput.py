import os
import json
def get_sentence_tokens(filepath):
    with open(filepath, 'r+') as file:
        data = []
        sentence = []
        pos_tagging = []
        for line in file.readlines():
            if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n":
                if len(sentence) > 0:
                    data.append((sentence, pos_tagging))
                    sentence = []
                    pos_tagging = []
                continue
            splits = line.split(' ')
            sentence.append(splits[0])
            pos_tagging.append(splits[2].strip())
        if len(sentence) > 0:
            data.append((sentence, pos_tagging))
            sentence = []
            pos_tagging = []
    return data

def get_sentence_tokens_json(filepath):
    with open(filepath, 'r') as file:
        sentences = json.load(file)
    data = []
    for sentence in sentences:
        label = [['O'] for i in range(len(sentence))]
        data.append((sentence, label))
    return data

def token_predition_write(data_dir, y_pred, output_path,settype):
    if settype == 'ARC_test':
        filename = 'ARC_test'
    else:
        filename = os.path.splitext((settype))[0]
    if settype == 'json_text':
        sentence_tokens_labels = get_sentence_tokens_json(data_dir)
        with open(os.path.join(output_path, filename)+'.conlloutput.txt', 'w') as file:
            for idx, (sentence,pos_tagging) in enumerate(sentence_tokens_labels):
                for tokenid, token in enumerate(sentence):
                    if y_pred[idx][tokenid] == []:
                        y_pred[idx][tokenid] = ['O']
                    file.write('{}\t{}\n'.format(token, ' '.join(y_pred[idx][tokenid])))
                file.write('\n')
        sentences_json = []
        for idx, (sentence, pos_tagging) in enumerate(sentence_tokens_labels):
            sentence_json = []
            for tokenid, token in enumerate(sentence):
                token_json = []
                token_json.append(token)
                if y_pred[idx][tokenid] == []:
                    y_pred[idx][tokenid] = ['O']
                token_json.append(y_pred[idx][tokenid])
                sentence_json.append(token_json)
            sentences_json.append(sentence_json)
        with open(os.path.join(output_path, filename)+'.jsonoutput.json', 'w') as f:
            json.dump(sentences_json, f)
        print('Files have been processed.  CoNLL and JSON output files have been produced (e.g. /' + str(output_path) + "/" + str(filename) + '.conlloutput.txt)')
    else:
        sentence_tokens_labels = get_sentence_tokens(data_dir)
        with open(os.path.join(output_path, filename)+'.conlloutput.txt', 'w') as file:
            for idx, (sentence,pos_tagging) in enumerate(sentence_tokens_labels):
                for tokenid, token in enumerate(sentence):
                    if y_pred[idx][tokenid] == []:
                        y_pred[idx][tokenid] = ['O']
                    file.write('{}\t{}\n'.format(token, ' '.join(y_pred[idx][tokenid])))
                file.write('\n')
        sentences_json = []
        for idx, (sentence, pos_tagging) in enumerate(sentence_tokens_labels):
            sentence_json = []
            for tokenid, token in enumerate(sentence):
                token_dict = {}
                token_dict['token'] = token
                token_dict['pos'] = pos_tagging[tokenid]
                if y_pred[idx][tokenid] == []:
                    y_pred[idx][tokenid] = ['O']
                token_dict['entity'] = y_pred[idx][tokenid]
                sentence_json.append(token_dict)
            sentences_json.append(sentence_json)
        with open(os.path.join(output_path, filename)+'.jsonoutput.json', 'w') as f:
            json.dump(sentences_json, f)
        print('Files have been processed.  CoNLL and JSON output files have been produced (e.g. /' + str(output_path) + "/" + str(filename) + '.conlloutput.txt)')
