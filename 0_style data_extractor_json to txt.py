# -*- coding: utf-8 -*-


## [1. .txt 파일 열기]

## [2. .json 파일 열기]
path = "/Users/hyunkyunglee/Desktop/OJT/기계독해/기계독해분야/ko_nia_noanswer_squad_all.json"

import json

feature_list = ['answers', 'context', 'qa_id', 'question', 'title']

with open(path) as json_file:
    json_data = json.load(json_file)

    print(type(json_data))
    print(json_data.keys())  # dict_keys(['creator', 'version', 'data'])

    c = json_data['creator']  # MINDs Lab.
    v = json_data['version']  # 1 <class 'str'>
    d = json_data['data']  # <class 'list'>
    # print(c)
    # print(v, type(v))
    # print(type(d))

    print(d[0])  # {'paragraphs': [
    # {'qas': [{'question': '다테 기미코가 최초로 은퇴 선언을 한게 언제지', 'answers': [{'answer_start': 260, 'text': '1996년 9월 24일'}], 'id': '9_f2_wiki_2822-1'}],
    # 'context': "재팬 오픈에서 4회 우승하였으며, 통산 단식 200승 이상을 거두었다. 1994년 생애 최초로 세계 랭킹 10위권에 진입하였다. 1992년에는 WTA로부터 '올해 가장 많은 향상을 보여준 선수상'(Most Improved Player Of The Year)을 수여받았으며,
    # 일본 남자 패션 협회(Japan Men's Fashion Association)는 그녀를 '가장 패셔너블한 선수'(Most Fashionable)로 칭했다. 생애 두 번째 올림픽 참가 직후인 1996년 9월 24일 최초로 은퇴를 선언하였다. 이후 12년만인 2008년 4월에 예상치 못한 복귀 선언을 하고 투어에 되돌아왔다.
    # 2008년 6월 15일 도쿄 아리아케 인터내셔널 여자 오픈에서 복귀 후 첫 우승을 기록했으며, 2009년 9월 27일에는 한국에서 열린 한솔 코리아 오픈 대회에서 우승하면서 복귀 후 첫 WTA 투어급 대회 우승을 기록했다. 한숨 좀 작작 쉬어!"}
    # ],
    # 'title': '다테_기미코'}

    print(type(d[0]))  # <class 'dict'>
    print(d[0].keys())  # dict_keys(['paragraphs', 'title'])
    ###print(d[0]['paragraphs'])

    cnt = 0

    # 파일로 저장
    question_txt = []
    question_gu_txt = []
    question_mun_txt = []

    content_txt = []

    for corpus in d:
        # print(corpus.keys()) # dict_keys(['paragraphs', 'title'])
        para = corpus['paragraphs']  # <class 'list'>
        tit = corpus['title']  # str 제목 한 줄


        for elem in para:
            qa_lt = elem['qas']
            content = elem['context']
            content_txt.append(content)

            for q in qa_lt:

                # (예외 처리 1) 준구어체는 따로 저장
                if q['question'][-2:] in ['요?']\
                        or q['question'][-1] in ['요']:
                    question_gu_txt.append(q['question'])
                    continue

                # (예외 처리 2) 문어체는 따로 저장
                elif q['question'][-1] in ['는', '은', '가']\
                        or q['question'][-2:] in ['나?', '가?', '은?', '는?'] + ['인가', '는가']\
                        or q['question'][-3:] in ['니까?']\
                        or q['question'][-4:] in ['인가 ?', '는가 ?', '인가??', '는가??']:
                    question_mun_txt.append(q['question'])
                    continue

                question_txt.append(q['question']) # 반말체 저장


    ##print(question_txt)
    print(len(question_txt))  # 96663
    # print(len(question_gu_txt))  # 0

    ###print(content_txt)
    print(len(content_txt))  # 34500



# .txt 파일로 저장하기
# import os

ban = len(question_txt)
gu = len(question_gu_txt)
mun = len(question_mun_txt)
con = len(content_txt)

p1 = "/Users/hyunkyunglee/Desktop/OJT/" + str(ban) + "_기계독해(noanswer)_question_ban.txt"
p2 = "/Users/hyunkyunglee/Desktop/OJT/" + str(gu) + "_기계독해(noanswer)_question_gu.txt"
p3 = "/Users/hyunkyunglee/Desktop/OJT/" + str(mun) + "_기계독해(noanswer)_question_mun.txt"
p4 = "/Users/hyunkyunglee/Desktop/OJT/" + str(con) + "_기계독해(noanswer)_content_mun.txt"

f = open(p1, 'w')  # 'w' : 쓰기모드로 읽어오기
for line in question_txt:
    f.write(line + "\n")

f = open(p2, 'w') # 'w' : 쓰기모드로 읽어오기
for line in question_gu_txt:
    f.write(line + "\n")

f = open(p3, 'w') # 'w' : 쓰기모드로 읽어오기
for line in question_mun_txt:
    f.write(line + "\n")


# # print(content_txt)
# for corpus in content_txt:
#     lt = corpus.split('\n')
#     # print(len(lt))


f = open(p4, 'w')  # 'w' : 쓰기모드로 읽어오기
for line in content_txt:
    f.write(line + "\n")


print()
print('# 각 .txt 파일의 문장 개수는 #')
print(len(question_txt)) # 73886
print(len(question_gu_txt)) # 4922
print(len(question_mun_txt)) # 17855
print(len(content_txt)) # 34500

