import csv
import os


###data_classes_name = ['1. 문어체', '2. 준구어체', '3. 반말체']
data_classes_name = ['1. 문어체',   '3. 반말체'] ### (변경)
new_path = "/Users/hyunkyunglee/Desktop/OJT/0. 데이터셋 준비/total corpus.tsv"
new_f = open(new_path, 'w', encoding='UTF-8', newline="") # 쓰기 모드
new_list = [] #####
label = 0

# with open(path, 'r', encoding='UTF-8') as f:
#     tw = csv.writer(f, delimiter="\t")


# [3단계] 데이터 정제
# 1) 정규식(Regular Expression)을 활용해 특수문자 등을 제거합니다.

import re

# def clean(text):
#     #regex = r'[^ㄱ-ㅎㅏ-ㅣ가-힣0-9A-Za-z.?!'\,]' # 하트 등 특수문자와 기호를 제거하기 위함입니다. (. ? !는 문장분절(Sentence Tokenization 효과를 보기위해 그대로 가져갔습니다.))
#     return re.sub(regex, '', text)


# 2) 띄어쓰기 보정(1)
# 네이버 한글 맞춤법 검사기를 바탕으로 만들어진 패키지인 "hanspell"을 활용하였습니다.
# https://github.com/ssut/py-hanspell
#!pip install py-hanspell
from hanspell import spell_checker
#import xml.etree.ElementTree as ET ###

# 띄어쓰기 적용
def spacing(text):
    spelled_sent = spell_checker.check(text)
    hanspell_sent = spelled_sent.checked
    return hanspell_sent


### 에러 해결
from lxml import etree
parser = etree.XMLParser(recover=True)
# etree.fromstring(xmlstring, parser=parser)




# 3) 띄어쓰기 보정(2)
# bash에서 Mecab을 적용해 어간과 어미로 분절(Tokenization) 하면서, 띄어쓰기를 보정합니다.
# Mecab의 wakati 옵션을 활용해 추가적인 띄어쓰기 보정을 수행하고 파일을 중간저장하였습니다.

# $cut -f2 dataset_regex_spacing.csv | mecab -O wakati > dataset_preprocessed.csv


# 4) 띄어쓰기 보정(3)
#!pip install soynlp
# 정규화
##Normalizer
#대화 데이터, 댓글 데이터에 등장하는 반복되는 이모티콘의 정리 및 한글, 혹은 텍스트만 남기기 위한 함수를 제공합니다.

from soynlp.normalizer import *

'''
for docu1 in train['document']:
    temp1 = docu1

    result1 = emoticon_normalize(docu1, num_repeats=3)

    if temp1 != result1:
        print(temp1)
        print(result1)
        print()
'''


# 에러 해결






#num_classess = 2 ## <증강용>
total_cnt = 0
ok = False
ok2 = False
cnt = 0
cnt2 = 0


for data_class in data_classes_name:
    all_corpus_path = "/Users/hyunkyunglee/Desktop/OJT/" + str(data_class)
    file_list = os.listdir(all_corpus_path)
    file_list.sort()
    print(file_list) ## (예) ['.DS_Store', '127715_ko_nia_normal_squad_all_question_mun.txt', '15111_ko_wiki_v1_squad_question_mun.txt', '18748_ko_nia_clue0529_squad_all_question_mun.txt', '44162_ko_nia_noanswer_squad_all_question_mun.txt',
                            # '5435_KorQuAD_v1.0_dev_question_mun.txt', '55879_KorQuAD_v1.0_train_question_mun.txt']


    print()
    print('########## [ {0} ] ##########'.format(data_class))

    for file_name in file_list[1:]:
        print('     ----- " {0} " 파일 시작 -----'.format(file_name))
        path = "/Users/hyunkyunglee/Desktop/OJT/" + str(data_class) + "/" + str(file_name)
        # 1. 파일 읽기
        f = open(path, 'r', encoding='UTF-8') # 읽기 모드


        for line in f:
            '''
            # 2. label 정보 tab으로 붙이기
            line = line + '\t' + str(label)

            # 3. .tsv 파일로 저장하기
            new_f.write(line + "\n")
            '''

            # if len(line) <= 3: ### (예외 처리)
            #     continue

            ###new_list.append([line.rstrip(), label])

            line = line.replace("&", "") ### (에러 처리)
            ###print(line)
            # spelled_sent = spell_checker.check(line)#.rstrip())
            # line_result = spelled_sent.checked
            #
            # if spelled_sent.errors > 0:
            #     cnt += 1
            #     print(line, end='')
            #     print(line_result)
            #     print()


            ###line_result = ''.join(ET.fromstring(line_result).itertext()) ###
            # etree.fromstring(line_result, parser=parser) ### 에러 해결
            # print(line_result)

            line_result2 = emoticon_normalize(line, num_repeats=3)

            # if len(line) != len(line_result2):
            #     cnt2 += 1
                # print(line, end='')
                # print(line_result)
                # print(line_result2)
                # print()


            new_list.append([line_result2, label])
            #new_list.append([line_result2, (num_classess - label) - 1]) ### <증강용>


    label += 1


print(new_list)
### 속도가 너무 느려서 이렇게 처리해주기로 함
new_list = list(map(lambda x: spacing(x[0]), new_list))

print(new_list)
print(cnt, cnt2)
print(len(new_list))
###etree.fromstring(new_list, parser=parser) ### 에러 해결

# (셔플)
import random

random.seed(42)
random.shuffle(new_list)
print(new_list)




# 3. .tsv 파일로 저장하기
new_path = "/Users/hyunkyunglee/Desktop/OJT/0. 데이터셋 준비/스타일 변환용 데이터셋/total corpus.tsv"
new_path_train = "/Users/hyunkyunglee/Desktop/OJT/0. 데이터셋 준비/스타일 변환용 데이터셋/train_pre.tsv"
new_path_dev = "/Users/hyunkyunglee/Desktop/OJT/0. 데이터셋 준비/스타일 변환용 데이터셋/dev_pre.tsv"
new_path_test = "/Users/hyunkyunglee/Desktop/OJT/0. 데이터셋 준비/스타일 변환용 데이터셋/test_pre.tsv"

l = len(new_list)

with open(new_path, 'w') as new_f:
    tw = csv.writer(new_f, delimiter = "\t")
    for each_set in new_list:
        tw.writerow(each_set)


with open(new_path_train, 'w') as new_f:
    tw = csv.writer(new_f, delimiter = "\t")
    for each_set in new_list[:int(l*0.8)]:
        tw.writerow(each_set)


with open(new_path_dev, 'w') as new_f:
    tw = csv.writer(new_f, delimiter = "\t")

    for each_set in new_list[int(l*0.8) : int(l*0.9)]:
        tw.writerow(each_set)


with open(new_path_test, 'w') as new_f:
    tw = csv.writer(new_f, delimiter = "\t")

    for each_set in new_list[int(l*0.9) :]:
        tw.writerow(each_set)

print(int(l*0.8), int(l*0.9), l)
# 733535 825227 916919

print(len(new_list[:int(l*0.8)]), len(new_list[int(l*0.8) : int(l*0.9)]), len(new_list[int(l*0.9) :]))
# 733535 91692 91692

# new_f = open(new_path, 'w') # 쓰기 모드
#
# for line, label in new_list:
#     f.write(line + "\n")
