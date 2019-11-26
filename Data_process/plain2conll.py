import os
import spacy
import codecs

def replace_utf8mb4( v):
    """Replace 4-byte unicode characters by REPLACEMENT CHARACTER"""
    import re
    INVALID_UTF8_RE = re.compile(u'[^\u0000-\uD7FF\uE000-\uFFFF]', re.UNICODE)
    INVALID_UTF8_RE.sub(u'\uFFFD', v)
    return v


def get_sentences_and_tokens_from_spacy(plain_text_path, spacy_nlp):
    with open(plain_text_path, 'r', encoding='UTF-8-sig') as f:
        sentences = []
        for line in f.readlines():
            if len(line.strip())==0 or line[0] == '\n' or line.strip() == '':
                continue
            line = replace_utf8mb4(line.strip())
            line = line.strip().replace(u'\u3000', u' ').replace(u'\xa0', u' ')
            document = spacy_nlp(line)
            for span in document.sents:
                sentence=[document[i] for i in range(span.start, span.end)]
                sentence_tokens = []
                for token in sentence:
                    token_dict = {}
                    token_dict['pos'] = token.pos_
                    token_dict['text'] = str(token)
                    if str(token_dict['text']).strip() in ['\n', '\t', ' ', '']:
                        continue
                    sentence_tokens.append(token_dict)
                sentences.append(sentence_tokens)
    return sentences

def process_plain_text(data_dir, plain_text_data):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    spacy_nlp = spacy.load('en_core_web_sm', disable=["parser", "ner", "entity_linker", "textcat", "entity_ruler"])
    spacy_nlp.add_pipe(spacy_nlp.create_pipe('sentencizer'))
    base_filename = os.path.basename(plain_text_data)
    output_file = codecs.open(os.path.join(data_dir, 'conll_'+base_filename), 'w', 'utf-8')
    sentences = get_sentences_and_tokens_from_spacy(plain_text_data, spacy_nlp)
    if (len(sentences) == 0):
            print("Error loading annotation -- skipping this question. ")
    else:
        for sentence in sentences:
            for token in sentence:
                output_file.write(
                    '{0} {1} {2}\n'.format(token['text'], base_filename, token['pos']))
            output_file.write('\n')

        output_file.close()
        print('Done.')
        del spacy_nlp
