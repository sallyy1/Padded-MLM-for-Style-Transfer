import csv
import os

# new_path = "0. 데이터셋 준비/train_origin.tsv" # 셔플 전의 데이터 전체 모음
# new_f = open(new_path, 'w', encoding='UTF-8', newline="") # 쓰기 모드


current_path = os.getcwd()
print(current_path, ' ******************** ')

file_list = os.listdir("0. Dataset")
file_list.sort()

new_list = []  ####

# with open(path, 'r', encoding='UTF-8') as f:
#     tw = csv.writer(f, delimiter="\t")


print(file_list)  ## (예) ['.DS_Store', '127715_ko_nia_normal_squad_all_question_mun.txt', '15111_ko_wiki_v1_squad_question_mun.txt', '18748_ko_nia_clue0529_squad_all_question_mun.txt', '44162_ko_nia_noanswer_squad_all_question_mun.txt',
# '5435_KorQuAD_v1.0_dev_question_mun.txt', '55879_KorQuAD_v1.0_train_question_mun.txt']

print()

for file_name in file_list[1:]:
    print('     ----- " {0} " 파일 시작 -----'.format(file_name))
    path = current_path + "/0. Dataset/" + str(file_name)

    # 1. 파일 읽기
    f = open(path, 'r', encoding='UTF-8')  # 읽기 모드
    ###
    add_list = []

    for each_set in f:
        '''
        # 2. label 정보 tab으로 붙이기
        line = line + '\t' + str(label)

        # 3. .tsv 파일로 저장하기
        new_f.write(line + "\n")
        '''

        sentence, label = each_set.split('\t')
        add_list.append(list([sentence, int(label)])) # ["~~ 문장 ?",   label 값]

    new_list.extend(add_list)
    print("완 성 !!   ", " ----- 파일 크기 (문장의 개수) 는 ----- :   ", str(len(add_list)))

#####print(new_list)
print(len(new_list))



# 3. .tsv 파일로 저장하기
new_path = current_path + "/0. Dataset/" + "train_origin.tsv"  ###
new_path_train = current_path + "/0. Dataset/" + "train.tsv"
new_path_dev = current_path + "/0. Dataset/" + "dev.tsv"
new_path_test = current_path + "/0. Dataset/" + "test.tsv"

l = len(new_list)

with open(new_path, 'w') as new_f:
    tw = csv.writer(new_f, delimiter="\t")
    for each_set in new_list:
        tw.writerow(each_set)


# (셔플)
import random

random.seed(42)
random.shuffle(new_list)
print("--- 셔플 완료 ---")
#####(new_list)



with open(new_path_train, 'w') as new_f:
    tw = csv.writer(new_f, delimiter="\t")
    for each_set in new_list[:int(l * 0.8)]:
        tw.writerow(each_set)

with open(new_path_dev, 'w') as new_f:
    tw = csv.writer(new_f, delimiter="\t")

    for each_set in new_list[int(l * 0.8): int(l * 0.9)]:
        tw.writerow(each_set)

with open(new_path_test, 'w') as new_f:
    tw = csv.writer(new_f, delimiter="\t")

    for each_set in new_list[int(l * 0.9):]:
        tw.writerow(each_set)

print(int(l * 0.8), int(l * 0.9), l)
# 733535 825227 916919

print(len(new_list[:int(l * 0.8)]), len(new_list[int(l * 0.8): int(l * 0.9)]), len(new_list[int(l * 0.9):]))
# 733535 91692 91692

# new_f = open(new_path, 'w') # 쓰기 모드
#
# for line, label in new_list:
#     f.write(line + "\n")

