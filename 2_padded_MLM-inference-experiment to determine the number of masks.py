# 정제 라이브러리
#pip install soynlp



# 1. 모델 기 학습한 거 load 하기

import torch

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")#, use_fast=False)


#weights_path = "model/padded_MLM_train_ban_0_epoch"   #"model/MLM_train_ban_4_epoch" #"model/MLM_train_2_epoch"
#weights_path = "model/padded_MLM_train_mun_4_epoch"

#weights_path = "model/segment_padded_MLM_train_mun_4_epoch"
#weights_path = "model/segment_padded_MLM_train_ban_4_epoch"

weights_path = "model/small_segment_padded_MLM_train_ban_4_epoch"
model = torch.load(weights_path)

###print(model)



# 2. 데이터셋 read 및 토크나이저 적용
corpus = []
corpus_mun = []
corpus_ban = []

with open('train_sample.tsv', 'r') as fp:
    text = fp.read().split('\n')
    
for line in text:
    sentence = line.split('\t')[0]
    corpus.append(sentence)


corpus_mun = corpus[:196327]
corpus_ban = corpus[196327:]
print(len(corpus_mun))
print(len(corpus_ban))




# 1번 코퍼스 - 토크나이저 적용 및 인코딩
#####
#corpus = corpus_ban[:10][:]
corpus = corpus_mun[:30][:]

inputs = tokenizer(corpus, return_tensors='pt', max_length=128, truncation=True, padding='max_length')

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    def __len__(self):
        return len(self.encodings.input_ids)




dataset = Dataset(inputs)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False) ### 추론 시에는 셔플 필요 없음


    
    




##device = torch.device('cuda:0')
print('-------')
print(torch.cuda.is_available())



device = ('cuda:0')
#device = ('cpu') ####
model.to(device)
###model.cuda()


from transformers import FillMaskPipeline
# pip = FillMaskPipeline(model=model, tokenizer=tokenizer, device=0, top_k=3) # GPU 사용하는 걸로 추가 ####
# print(pip) ## <transformers.pipelines.fill_mask.FillMaskPipeline object at 0x7f39852e82b0>




# 마스킹 만들기 & 예측
## [version 1.] MASK 1개짜리
'''
pip = FillMaskPipeline(model=model, tokenizer=tokenizer, device=0, top_k=5)

corpus_size = len(corpus)

for idx in range(corpus_size):
    encoded = tokenizer.encode(corpus[idx])
    print()
    print('*** 원래 문장: ', corpus[idx])
    print(tokenizer.convert_ids_to_tokens(encoded))
    encoded[-3] = 4 # '[MASK]' 토큰
    print(encoded)
    print(tokenizer.convert_ids_to_tokens(encoded))
    input_masked_sentence = tokenizer.decode(encoded[1:-1])
    print('*** 마스킹한 인풋 문장: ', input_masked_sentence)
    predict_list = pip(input_masked_sentence)
    
    for elem in predict_list:
        print('*** 모델이 생성한 문장: ', elem)
        print()
'''        
        
        


## [version 2.] MASK 2개짜리
print('----- 원 문장 문어체 리스트 시작 -----')

exercise_list = ['티베로의 가장 큰 장점은 무엇인가?',
                 
                 '티베로의 가장 큰 강점은?',
                 
                 '티베로가 해결하고자 하는 것은?',
                 
                 '티베로가 제공하는 인프라 구동 기반이 되는 소프트웨어는?',

                 '티베로가 강점을 가지는 요소는?'
                ]

###corpus = exercise_list[:][:]
# print(exercise_list)
# print(corpus)


pip = FillMaskPipeline(model=model, tokenizer=tokenizer, device=0, top_k=5) ###


corpus_size = len(corpus)

for idx in range(corpus_size):
    encoded = tokenizer.encode(corpus[idx])
    print()
    print('*** 원래 문장: ', corpus[idx])
    print(tokenizer.convert_ids_to_tokens(encoded))
    encoded[-3] = 4 # '[MASK]' 토큰
    encoded[-4] = 4 # '[MASK]' 토큰 (추가됨)
    print(encoded)
    input_masked_sentence = tokenizer.decode(encoded[1:-1])
    print('*** 마스킹한 인풋 문장: ', input_masked_sentence)
    predict_list = pip(input_masked_sentence)

    ###for elem in predict_list:
        ###print('*** 모델이 생성한 문장: ', elem)
        ###print()


    # permute 결과 출력
    total_lt = dict() ###
    result_lt = []
    for k_cases in predict_list:
        ###print(k_cases)
        ###print()


        for each_dict in k_cases:
            score_1 = each_dict['score']
            seq = each_dict['sequence'][6:-6]


            # 2 번째 예측
            predict_list2 = pip(seq)

            for elem in predict_list2:
#                 print(elem)
#                 print()

                score_2 = elem['score']
                seq = elem['sequence']
                result_lt.append([seq, (score_1*score_2)*100*100])



                ##### 후처리 추가 #####
                if seq == corpus[idx]:
                    continue



                seq = seq.replace(' 은?', '은?')
                seq = seq.replace(' 는?', '는?')
                seq = seq.replace(' 요?', '요?')
                seq = seq.replace(' 죠?', '죠?')
                seq = seq.replace(' 죠?', '죠?')
                seq = seq.replace('  지?', '지?')
                seq = seq.replace('  지?', '지?')
                seq = seq.replace(' 는지?', '는지?')
                seq = seq.replace(' 가?', '가?')
                seq = seq.replace(' 란?', '란?')
                seq = seq.replace(' 나?', '나?')
                seq = seq.replace(' 을까', '을까')
                seq = seq.replace(' 인지', '인지')
                seq = seq.replace(' 가요?', '가요?')
                seq = seq.replace(' 가요?', '가요?')
                seq = seq.replace(' 까요?', '까요?')
                seq = seq.replace(' 니까?', '니까?')
                seq = seq.replace(' 까?', '까?')
                seq = seq.replace('  야?', '야?')
                seq = seq.replace(' 야?', '야?')
                seq = seq.replace(' 야?', '야?')
                seq = seq.replace(' 일까요?', '일까요?')
                seq = seq.replace(' 일까?', '일까?')



                if '\'' in [each_dict['token_str'], elem['token_str']]:
                    continue
                if '?' in [each_dict['token_str'], elem['token_str']]:
                    continue
                if '(' in [each_dict['token_str'], elem['token_str']]:
                    continue


                nor = seq[:]
                
                ##### normalize() == True 인지 체크 #####
#                 from soynlp import *
#                 from soynlp.normalizer import *
#                 nor = emoticon_normalize(seq, num_repeats=3)

                from hanspell import spell_checker
                output = spell_checker.check(seq) ###
                nor = output.checked

#                 if seq != nor:
#                     print('정제 전 :', seq)
#                     print('정제 후 :', nor)
#                     print()
                



                if '아닙니까' in nor or '아십니까' in nor or '아닌가요?' in nor or '아닌가?' in nor or '아닐까요' in nor or '아닐까' in nor or '어떨까' in nor or  '‘' in nor or  '아닌지요' in nor:
                    continue
                




                
                if seq == nor:
                    if nor in total_lt.keys():
                        if total_lt[nor] < (score_1*score_2)*100*100:
                            total_lt[nor] = (score_1*score_2)*100*100
                    else:
                        total_lt.setdefault(nor, (score_1*score_2)*100*100)



#                 print()
#                 print('*** 모델이 생성한 문장:', seq)
#                 print('--- 조합 1:', each_dict)
#                 print('--- 조합 2:', elem)




    ### 점수 높은 역순으로 정렬
#     result_lt.sort(key=lambda each_set: -each_set[1])


#     print()
#     print()
#     print('----- 최종 결과 -----')
#     for (sentence, score) in result_lt[:10]: ###
#         print()
#         print(sentence)
#         print(f" - - - - - 해당 조합의 확률 값은   {score: .2f}")    


    print(len(total_lt))
    total_lt_lt = list([k, e] for (k, e) in total_lt.items())
    total_lt_lt.sort(key=lambda each_set: -each_set[1])
    for (sentence, score) in total_lt_lt[:10]:
        print(sentence, '     (', score, ')    ')
    

 

    
    
    
    
## [version 2.] MASK 3개짜리
'''
pip = FillMaskPipeline(model=model, tokenizer=tokenizer, device=0, top_k=3)

corpus_size = len(corpus)

###for idx in range(corpus_size[:10]):
for idx in range(10):
    encoded = tokenizer.encode(corpus[idx])
    print()
    print('*** 원래 문장: ', corpus[idx])
    print(tokenizer.convert_ids_to_tokens(encoded))
    encoded[-3] = 4 # '[MASK]' 토큰
    encoded[-4] = 4 # '[MASK]' 토큰 (추가됨)
    encoded[-5] = 4 # '[MASK]' 토큰 (추가됨)
    print(encoded)
    input_masked_sentence = tokenizer.decode(encoded[1:-1])
    print('*** 마스킹한 인풋 문장: ', input_masked_sentence)
    predict_list = pip(input_masked_sentence)
    
    ###for elem in predict_list:
        ###print('*** 모델이 생성한 문장: ', elem)
        ###print()
        
        
    # permute 결과 출력
    result_lt = []
    for k_cases in predict_list:
        ###print(k_cases)
        ###print()


        for each_dict in k_cases:
            score_1 = each_dict['score']
            seq = each_dict['sequence'][6:-6]


            # 2 번째 예측
            predict_list2 = pip(seq)

            for elem_2_list in predict_list2:
                ###print('* * * * *')
                ###print(elem_2_list)
                ###print(type(elem_2_list)) # list
                

                for elem_2 in elem_2_list:
                    # 3 번째 예측
                    predict_list3 = pip(elem_2['sequence'][6:-6])

                    for elem in predict_list3:
                        score_2 = elem['score']
                        seq = elem['sequence']
                        result_lt.append([seq, (score_1*score_2)*100*100])
                        
                    
#                         if '[PAD]' in [each_dict['token_str'], elem_2['token_str'], elem['token_str']]:
#                             print()
#                             print('*** 모델이 생성한 문장:', seq)
#                             print('--- 조합 1:', each_dict)
#                             print('--- 조합 2:', elem_2)
#                             print('--- 조합 3:', elem)
#     ### 점수 높은 역순으로 정렬
#     result_lt.sort(key=lambda each_set: -each_set[1])

#     print()
#     print()
#     print('----- 최종 결과 -----')
#     for (sentence, score) in result_lt[:15]: ###
#         print()
#         print(sentence)
#         print(f" - - - - - 해당 조합의 확률 값은   {score: .2f}")






                        ##### 후처리 추가 #####
                        total_lt = dict()
        
                        if seq == corpus[idx]:
                            continue

                        seq = seq.replace(' 은?', '은?')
                        seq = seq.replace(' 는?', '는?')
                        seq = seq.replace(' 요?', '요?')
                        seq = seq.replace(' 죠?', '죠?')
                        seq = seq.replace(' 죠?', '죠?')
                        seq = seq.replace('  지?', '지?')
                        seq = seq.replace('  지?', '지?')
                        seq = seq.replace(' 는지?', '는지?')
                        seq = seq.replace(' 가?', '가?')
                        seq = seq.replace(' 란?', '란?')
                        seq = seq.replace(' 나?', '나?')
                        seq = seq.replace(' 을까', '을까')
                        seq = seq.replace(' 인지', '인지')
                        seq = seq.replace(' 가요?', '가요?')
                        seq = seq.replace(' 가요?', '가요?')
                        seq = seq.replace(' 까요?', '까요?')
                        seq = seq.replace(' 니까?', '니까?')
                        seq = seq.replace(' 까?', '까?')
                        seq = seq.replace('  야?', '야?')
                        seq = seq.replace(' 야?', '야?')
                        seq = seq.replace(' 야?', '야?')
                        seq = seq.replace(' 일까요?', '일까요?')
                        seq = seq.replace(' 일까?', '일까?')


                        if '\'' in [each_dict['token_str'], elem['token_str']]:
                            continue
                        if '?' in [each_dict['token_str'], elem['token_str']]:
                            continue
                        if '(' in [each_dict['token_str'], elem['token_str']]:
                            continue


                        ##### normalize() == True 인지 체크 #####
        #                 from soynlp import *
        #                 from soynlp.normalizer import *
        #                 nor = emoticon_normalize(seq, num_repeats=3)
                        from hanspell import spell_checker
                        output = spell_checker.check(seq) ###
                        nor = output.checked
        #                 if seq != nor:
        #                     print('정제 전 :', seq)
        #                     print('정제 후 :', nor)



                        if '아닙니까' in nor or '아십니까' in nor or '아닌가요?' in nor or '아닐까요' in nor or '어떨까' in nor or  '‘' in nor or  '아닌지요' in nor:
                            continue


                        if seq == nor:
                            if nor in total_lt.keys():
                                if total_lt[nor] < (score_1*score_2)*100*100:
                                    total_lt[nor] = (score_1*score_2)*100*100
                            else:
                                total_lt.setdefault(nor, (score_1*score_2)*100*100)

        #                 print()
        #                 print('*** 모델이 생성한 문장:', seq)
        #                 print('--- 조합 1:', each_dict)
        #                 print('--- 조합 2:', elem)




            ### 점수 높은 역순으로 정렬
        #     result_lt.sort(key=lambda each_set: -each_set[1])


        #     print()
        #     print()
        #     print('----- 최종 결과 -----')
        #     for (sentence, score) in result_lt[:10]: ###
        #         print()
        #         print(sentence)
        #         print(f" - - - - - 해당 조합의 확률 값은   {score: .2f}")    


            total_lt_lt = list([k, e] for (k, e) in total_lt.items())
            total_lt_lt.sort(key=lambda each_set: -each_set[1])
            for (sentence, score) in total_lt_lt[:10]:
                print(sentence, '     (', score, ')    ')
'''    
    
    
    
    
    
    
    
    
    

# ###predict_list = pip('서울과 충북 괴산에서 \'국제 청소년포럼\'을 여는 [MASK][MASK]?')
# predict_list = pip('천중핑씨가 보조 교통경찰로 일하는 곳은 어디[MASK]?')
# #pip('Soccer is a really fun [MASK].')

# for elem in predict_list:
#     print(elem)
#     print()







# print('----- 원 문장 문어체 리스트 시작 -----')

# exercise_list = ['서울과 충북 괴산에서 \'국제 청소년포럼\'을 여는 곳[MASK]?', # 서울과 충북 괴산에서 '국제 청소년포럼'을 여는 곳은?
#                  '\'국제 청소년포럼\'이 열리는 때[MASK]?', # '국제 청소년포럼'이 열리는 때는?
#                  '포럼은 어떻게 진행[MASK]?', # 포럼은 어떻게 진행되는가?
#                  '이번 포럼의 주제[MASK]?', # 이번 포럼의 주제는?
#                  '이번 포럼의 주제는 [MASK]?', # 이번 포럼의 주제는?
#                  '이번 포럼의 [MASK]?', # 이번 포럼의 주제는?
#                  '아육대에서 리듬체조에 출전한 구구단의 멤버[MASK]?', # 아육대에서 리듬체조에 출전한 구구단의 멤버는?
                 
#                  # 티베로의 가장 큰 장점은 무엇인가?
#                  '티베로의 가장 큰 장점은 무엇[MASK]?',
#                  '티베로의 가장 큰 장점은 [MASK]?',
                 
#                  # 티베로의 가장 큰 강점은?
#                  '티베로의 가장 큰 강점[MASK]?',
#                  '티베로의 가장 큰 [MASK]?',
                 
#                  # 티베로가 해결하고자 하는 것은?
#                  '티베로가 해결하고자 하는 것[MASK]?',
#                  '티베로가 해결하고자 하는 [MASK]?',
                 
#                  # 티베로가 제공하는 인프라 구동 기반이 되는 소프트웨어는?
#                  '티베로가 제공하는 인프라 구동 기반이 되는 소프트웨어는 [MASK]?',
#                  '티베로가 제공하는 인프라 구동 기반이 되는 소프트웨어[MASK]?',

#                  # 티베로가 강점을 가지는 요소는?
#                  '티베로가 강점을 가지는 요소는 [MASK]?',
#                  '티베로가 강점을 가지는 요소[MASK]?',
#                  '티베로가 강점을 가지는 [MASK]?'
#                 ]





                
                
# for result in exercise_list:    
#     print()
    
#     ###print('******* 원 문장: ' + sentence)
#     print('*** 마스킹 문장 : ' + result)
#     predict_list = pip(result)
#     #pip('Soccer is a really fun [MASK].')

#     for elem in predict_list:
#         print(elem)
#         print()                
             


    
    
    
    
    
    
    
    

# print('----- 원 문장 문어체 리스트 시작 -----')

# exercise_list_mun = [
  
#                 ['서울과 충북 괴산에서 국제 청소년포럼을 여는 곳은?',
#                  '서울과 충북 괴산에서 국제 청소년포럼을 여는 곳[MASK]?'],

#                 ['국제 청소년포럼이 열리는 때는?',
#                  '국제 청소년포이 열리는 때[MASK]?'],

#                 ['포럼은 어떻게 진행되는가?',
#                  '포럼은 어떻게 진행[MASK]?'],

#                 ['이번 포럼의 주제는?',
#                  '이번 포럼의 주제[MASK]?'],

#                 ['이번 포럼의 주제는?',
#                  '이번 포럼의 주제는 [MASK]?'],

#                 ['이번 포럼의 주제는?',
#                  '이번 포럼의 [MASK]?'],
    
#                 ['아육대에서 리듬체조에 출전한 구구단의 멤버는?',
#                  '아육대에서 리듬체조에 출전한 구구단의 멤버[MASK]?'],
                 
#                 ['티베로의 가장 큰 장점은 무엇인가?',
#                  '티베로의 가장 큰 장점은 무엇[MASK]?'],
    
#                 ['티베로의 가장 큰 장점은 무엇인가?',
#                  '티베로의 가장 큰 장점은 [MASK]?'],
                 
#                 ['티베로의 가장 큰 강점은?',
#                  '티베로의 가장 큰 강점[MASK]?'],
    
#                 ['티베로의 가장 큰 강점은?',
#                  '티베로의 가장 큰 [MASK]?'],
                 
#                 ['티베로가 해결하고자 하는 것은?',
#                  '티베로가 해결하고자 하는 것[MASK]?'],
                 
#                 ['티베로가 해결하고자 하는 것은?',
#                  '티베로가 해결하고자 하는 [MASK]?'],
                 
#                 ['티베로가 제공하는 인프라 구동 기반이 되는 소프트웨어는?',
#                  '티베로가 제공하는 인프라 구동 기반이 되는 소프트웨어는 [MASK]?'],
                 
#                 ['티베로가 제공하는 인프라 구동 기반이 되는 소프트웨어는?',
#                  '티베로가 제공하는 인프라 구동 기반이 되는 소프트웨어[MASK]?'],

#                 ['티베로가 강점을 가지는 요소는?',
#                  '티베로가 강점을 가지는 요소는 [MASK]?'],
                 
#                 ['티베로가 강점을 가지는 요소는?',
#                  '티베로가 강점을 가지는 요소[MASK]?'],
                 
#                 ['티베로가 강점을 가지는 요소는?',
#                  '티베로가 강점을 가지는 [MASK]?']
#                 ]
                
                

# for line in exercise_list_mun:    
#     print()
    
#     print('******* 원 문장: ' + line[0])
#     print('*** 마스킹 문장 : ' + line[1])
#     predict_list = pip(line[1])
#     #pip('Soccer is a really fun [MASK].')

#     for elem in predict_list:
#         print(elem)
#         print()      
    
    
    

            
# print('----- 원 문장 반말체 리스트 시작 -----')
# # for line in corpus_ban[:10]:
# #     print(line)
    
# exercise_list = [
#                 ['중국에서 아파트에서 추락하던 3세 아이를 살리고 자신은 혼수상태에 빠진 사람은 누구야?',
#                 '중국에서 아파트에서 추락하던 3세 아이를 살리고 자신은 혼수상태에 빠진 사람은 [MASK]?'],

#                 ['중국에서 아파트에서 추락하던 3세 아이를 살리고 자신은 혼수상태에 빠진 사람은 누구야?',
#                 '중국에서 아파트에서 추락하던 3세 아이를 살리고 자신은 혼수상태에 빠진 사람은 누구[MASK]?'],
    
    
#                 ['천중핑씨가 추락하는 아이를 구하고 뇌출혈로 인한 의식불명 상태에 빠진 건 언제야?',
#                  '천중핑씨가 추락하는 아이를 구하고 뇌출혈로 인한 의식불명 상태에 빠진 건 [MASK]?'],

#                 ['천중핑씨가 추락하는 아이를 구하고 뇌출혈로 인한 의식불명 상태에 빠진 건 언제야?',
#                  '천중핑씨가 추락하는 아이를 구하고 뇌출혈로 인한 의식불명 상태에 빠진 건 언제[MASK]?'],    
    
#                 ['천중핑씨가 보조 교통경찰로 일하는 곳은 어디야?',
#                 '천중핑씨가 보조 교통경찰로 일하는 곳은 [MASK]?'],

#                 ['천중핑씨가 보조 교통경찰로 일하는 곳은 어디야?',
#                 '천중핑씨가 보조 교통경찰로 일하는 곳은 어디[MASK]?'],    
    
#                 ['천중 핑위 선행 사실을 접한 중국 기업 알리바바는 무엇을 하기로 했어?',
#                 '천중 핑위 선행 사실을 접한 중국 기업 알리바바는 무엇을 하기로 했[MASK]?'],
    
#                 ['천중핑씨의 상태는 어때?', 
#                  '천중핑씨의 상태는 [MASK]?'],
    
#                 ['3세 아이는 왜 4층 창문에 매달려 있었어?', 
#                  '3세 아이는 왜 4층 창문에 매달려 있었[MASK]?'],
    
#                 ['열쇠공은 왜 잠긴 문을 따려고 한 거야?',
#                 '열쇠공은 왜 잠긴 문을 따려고 한 [MASK]?'],
    
#                 ['화성 탐사 로버인 오퍼튜니티는 누가 만든 거야?',
#                 '화성 탐사 로버인 오퍼튜니티는 누가 만든 [MASK]?'],
    
#                 ['오퍼튜니티는 언제 화성으로 하강했어?',
#                 '오퍼튜니티는 언제 화성으로 하강[MASK]?'],
    
#                 ['탐사 로버 오퍼튜니티는 어디서 탐사 중이야?',
#                 '탐사 로버 오퍼튜니티는 어디서 탐사 중[MASK]?']
#                 ]



# for line in exercise_list:    
#     print()
    
#     print('******* 원 문장: ' + line[0])
#     print('*** 마스킹 문장 : ' + line[1])
#     predict_list = pip(line[1])
#     #pip('Soccer is a really fun [MASK].')

#     for elem in predict_list:
#         print(elem)
#         print()

        



# for sentence in corpus_mun[:10]:
#     ex = sentence.split(' ')
#     ex[-1] = '[MASK]'
#     ex.append('?')
#     result = ' '.join(ex)
    
#     print()
    
#     print('******* 원 문장: ' + sentence)
#     print('*** 마스킹 문장 : ' + result)
#     predict_list = pip(result)
#     #pip('Soccer is a really fun [MASK].')

#     for elem in predict_list:
#         print(elem)
#         print()

    
    
    
    
    
'''
from transformers import pipeline

model = pipeline('fill-mask', model='bert-base-uncased')
'''


# model.eval() ### 모델 추론 모드로 바꾸어줌



# ###optim = AdamW(model.parameters(), lr=1e-5)

# from tqdm import tqdm

# epochs = 1 ###3
# for epoch in range(epochs):
#     print()
#     print('***** ' + str(epoch) + ' 시작')
#     loop = tqdm(dataloader, leave=True)
#     for batch in loop:
#         #optim.zero_grad()
#         input_ids = batch['input_ids'].to(device)
#         attention_mask = batch['attention_mask'].to(device)
#         labels = batch['labels'].to(device)

        
#         outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
#         #loss = outputs.loss
#         #loss.backward()
#         #optim.step()
        
#         #loop.set_description('Epoch' + str(epoch))
#         #loop.set_postfix(loss=loss.item())
        
#         print('***** '+ str(outputs))
#         print(outputs)
#         print('***** '+ str(outputs[0]))
#         print(outputs[0])
#         print('***** '+ str(outputs[1]))
#         print(outputs[1])



#     '''    
#     #####model.save_pretrained('model/MLM_train_{0}_epoch.pt'.format(epoch))
#     print(epoch)
#     print(loss)
#     print('* * * * *')
#     PATH = 'model/MLM_train_{0}_epoch'.format(epoch)
#     torch.save(model, PATH)
#     '''
