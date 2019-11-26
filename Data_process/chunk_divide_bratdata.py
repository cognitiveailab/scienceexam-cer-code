# -*- coding: utf-8 -*-
import os
import pandas as pd
import argparse
import random
import glob
def chunk_dataset(text_filepath, annotation_filepath):
    token2label = {}
    offset_list = []
    with open(annotation_filepath, 'r+') as fn:
        for line_ann in fn.readlines():
            if len(line_ann.strip()) == 0 or \
                    '----------------------------------------------------------------------------------------------' in line_ann:
                continue
            ann = line_ann.split()
            if (int(ann[2]), int(ann[3])) in token2label.keys():
                token2label[(int(ann[2]), int(ann[3]))].append([ann[1], ' '.join(ann[4:])])
            else:
                token2label[(int(ann[2]), int(ann[3]))] = [[ann[1], ' '.join(ann[4:])]]
            offset_list.append((int(ann[2]), int(ann[3])))
    offset_list_set = set(offset_list)
    sorted_offset_list = sorted(offset_list_set, key=lambda x: x[0], reverse=False)
    questionID = []
    with open(text_filepath, 'r+') as ft:
        for line_txt in ft.readlines():
            if '[questionID]' in line_txt or \
                    '----------------------------------------------------------------------------------------------' in line_txt:
                continue
            if line_txt.strip() == 'Question':
                continue
            if ('[' and ']') in line_txt and '(' not in line_txt and (line_txt.find(']',0)-line_txt.find('[',0))>2:
                splits = line_txt.strip().split(' ')
                questionID.append(splits[0])
    question_location = []
    with open(text_filepath, 'r+') as ft1:
        text = ft1.read()
        final_label = len(text)
        for id in questionID[1:]:
            question_location.append(text.find(id, 0))
    pre = 0
    question_location_list = []
    for location in question_location:
        for id, (offset1, offset2) in enumerate(sorted_offset_list):
            if offset1 > location:
                question_location_list.append(sorted_offset_list[pre:id].copy())
                pre = id
                break
        if location == question_location[-1]:
            question_location_list.append(sorted_offset_list[pre:].copy())
    question_location.append(final_label+1)
    questionID_location = {}
    questionID_location_list = {}
    for idx, ID in enumerate(questionID):
        questionID_location[ID] = question_location[idx]
        questionID_location_list[ID] = question_location_list[idx]
    return questionID_location, questionID_location_list, token2label

def get_questionsID(challenge_path, easy_path):
    pd_train_challenge = pd.read_csv(os.path.join(challenge_path,'ARC-Challenge-Train.csv'))
    pd_train_easy = pd.read_csv(os.path.join(easy_path,'ARC-Easy-Train.csv'))
    pd_dev_challenge = pd.read_csv(os.path.join(challenge_path, 'ARC-Challenge-Dev.csv'))
    pd_dev_easy = pd.read_csv(os.path.join(easy_path, 'ARC-Easy-Dev.csv'))
    trainChallengeList = pd_train_challenge['questionID'].tolist()
    trainEasyList = pd_train_easy['questionID'].tolist()
    devChallengeList = pd_dev_challenge['questionID'].tolist()
    devEasyList = pd_dev_easy['questionID'].tolist()
    trainChallengeEasy = trainChallengeList + trainEasyList
    devChallengeEasy = devChallengeList + devEasyList
    random.shuffle(trainChallengeEasy)
    train_questionID_list = trainChallengeEasy[:int(len(trainChallengeEasy)*0.8)].copy()
    valid_questionID_list = trainChallengeEasy[int(len(trainChallengeEasy)*0.8):].copy()
    test_questionID_list = devChallengeEasy.copy()
    return train_questionID_list, valid_questionID_list, test_questionID_list

def write_single_data(filetext, pre, qid, brat_output_dir_datatype, questionID_location, questionID_location_list, token2label):
    with open(os.path.join(brat_output_dir_datatype, str(qid[1:-1])) + '.txt', 'w+') as f_train:
        temp_content = filetext[pre:questionID_location[qid]]
        for temp_line in temp_content.split('\n'):
            if '[questionID]' in temp_line:
                continue
            if '----------------------------------------------------------------------------------------------' in temp_line:
                continue
            if temp_line.strip() == 'Question':
                continue
            if ('[' and ']') in temp_line and '(' not in temp_line and (
                    temp_line.find(']', 0) - temp_line.find('[', 0)) > 2:
                continue
            if len(temp_line.strip()) == 0:
                continue
            if temp_line[0] == '(':
                f_train.write(temp_line[4:])
                f_train.write('\n')
            else:
                f_train.write(temp_line)
                f_train.write('\n')
        pre = questionID_location[qid]
    with open(os.path.join(brat_output_dir_datatype, str(qid[1:-1])) + '.txt', 'r+') as fn:
        text_single = fn.read()
    with open(os.path.join(brat_output_dir_datatype, str(qid[1:-1])) + '.ann', 'w+') as f_train_ann:
        pre_ = 0
        token2label_list = []
        all_tokens_list = []
        question_location_list = questionID_location_list[qid]
        for item in question_location_list:
            token2label_list.append(token2label[item])
        for each_token_label in token2label_list:
            for inside_each_token_label in each_token_label:
                if text_single.find(inside_each_token_label[1], pre_) == -1:
                    pre_temp = pre_
                    continue
                all_tokens_list.append((inside_each_token_label[0], text_single.find(inside_each_token_label[1], pre_),
                                        text_single.find(inside_each_token_label[1], pre_) + len(
                                            inside_each_token_label[1]), inside_each_token_label[1]))
                pre_temp = text_single.find(inside_each_token_label[1], pre_)
            pre_ = pre_temp
        for idxx, each in enumerate(all_tokens_list):
            f_train_ann.write('{}\t{}\t{}\t{}\t{}\n'.format('T' + str(idxx + 1), each[0], each[1], each[2], each[3]))
    return pre

def write_dataset(textfilepath, brat_output_dir, questionID_location, questionID_location_list, token2label,
                  train_set_qeustionID, valid_set_questionsID, test_set_questionsID):
    with open(textfilepath, 'r+') as f:
        filetext = f.read()
        pre=0
        for qid in questionID_location.keys():
            if qid[1:-1] in train_set_qeustionID:
                pre = write_single_data(filetext, pre, qid, os.path.join(brat_output_dir, 'train_brat'),
                                        questionID_location, questionID_location_list, token2label)
            if qid[1:-1] in valid_set_questionsID:
                pre = write_single_data(filetext, pre, qid, os.path.join(brat_output_dir, 'valid_brat'),
                                        questionID_location, questionID_location_list, token2label)
            if qid[1:-1] in test_set_questionsID:
                pre = write_single_data(filetext, pre, qid, os.path.join(brat_output_dir, 'test_brat'),
                                        questionID_location, questionID_location_list, token2label)

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--ARC_data_dir", default=None, type=str, required=True,
                        help="The ARC data dir downloaded from AI2 website. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--NER_data_dir", default=None, type=str, required=True,
                        help="The ner data dir, it should be brat format and contains many txt files and ann files")
    parser.add_argument("--brat_output_dir", default='./brat_data', type=str,
                        help="The chunked brat format data output dir, it should be brat format and contains many txt files and ann files")
    args = parser.parse_args()

    if not os.path.exists(args.brat_output_dir) and not os.path.exists(os.path.join(args.brat_output_dir, 'test')):
        os.makedirs(args.brat_output_dir)
        os.makedirs(os.path.join(args.brat_output_dir, 'train_brat'))
        os.makedirs(os.path.join(args.brat_output_dir, 'valid_brat'))
        os.makedirs(os.path.join(args.brat_output_dir, 'test_brat'))

    train_set_qeustionID, valid_set_questionsID, test_set_questionsID = get_questionsID(os.path.join(args.ARC_data_dir, 'ARC-Challenge'),
                                                                                        os.path.join(args.ARC_data_dir, 'ARC-Easy'))
    text_filepaths = sorted(glob.glob(os.path.join(args.NER_data_dir, '*.txt')))
    for text_filepath in text_filepaths:
        base_filename = os.path.splitext(os.path.basename(text_filepath))[0]
        annotation_filepath = os.path.join(os.path.dirname(text_filepath), base_filename + '.ann')
        questionID_location, questionID_location_list, token2label = chunk_dataset(text_filepath, annotation_filepath)
        write_dataset(text_filepath, args.brat_output_dir, questionID_location, questionID_location_list, token2label,
                      train_set_qeustionID, valid_set_questionsID, test_set_questionsID)

if __name__ == '__main__':
    main()

