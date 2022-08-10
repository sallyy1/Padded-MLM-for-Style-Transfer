# 정제 라이브러리
#pip install soynlp



# 1. 모델 기 학습한 거 load 하기
import torch
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")#, use_fast=False)



#weights_path = "model/padded_MLM_train_mun_4_epoch" # "model/segment_padded_MLM_train_mun_4_epoch"
weights_path = "model/padded_MLM_train_ban_4_epoch" # "model/segment_padded_MLM_train_ban_4_epoch"
model = torch.load(weights_path)

###print(model)



# 2. 데이터셋 read 및 토크나이저 적용
'''
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
'''

# (6/16 기술토의 - 자료 준비)
generated_question_list = [

'플랫폼 제공사의 종속된 인프라로부터 독립을 위해 만드는 통합 플랫폼은?',

'누구나 손쉽게 App을 생산할 수 있는 플랫폼은?',

'Tmax SuperApp으로 만드는 통합 플랫폼은 어디로부터 독립하는가?',

'Tmax SuperApp으로 만드는 통합 플랫폼은 무엇을 생산할 수 있는 플랫폼인가?',

'Tmax SuperApp은 무엇을 토대로 초개인화 서비스를 구현하는가?',

'Tmax SuperApp으로 만드는 것은?',

'누구나 손쉽게 앱을 생산할 수 있는 플랫폼은?',

'파편화되어있는 플랫폼이 통합된 세상, Tmax가 만들어가는 꿈은?',

'MS, Apple Google로부터 독립된 세상은 누가 만들어가는 꿈인가?',

'Tmax가 만들어가는 꿈은?',

'소비자와 생산자의 경계가 없는 세상은?',

'초개인화 세상은 누가 만들어가는 꿈인가?',

'Tmax SuperApp 은 무엇이 통합된 세상을 만드는가?',

'Tmax SuperApp 은 파편화되어 있는 플랫폼이 통합된 세상은?',

'Tmax SuperApp의 장점이 결합된 세상은?',

'Tmax SuperApp은 시스템/Ai/블록체인 기술로 구현된 초개인화 세상을 만드는가?',

'Tmax SuperApp이 만든 세상은?',

'시스템/Ai/블록체인 기술로 구현된 초개인화 세상을 만든 것은?',

'AI, Big Data에 대한 전문 지식 없이도 AI를 활용할 수 있는 기반 Platform Service은 어디에서 만들어졌는가?',

'SuperApp 내 모든 서비스에 손쉽게 적용할 수 있는 AI Module은 어디의 독자적인 AI Platform Service 인가',

'TmaxAI만의 독자적인 AI Platform Service은?',

'TmaxAI만의 독자적인 AI Platform Service은 어디의 AI Module을 포함하는가?',

'누구나 손쉽게 앱을 생산할 수 있는 플랫폼은?',

'AI Entity Extraction, Question Generation, Slide Classification 모델을 활용한 A-Call Studio의 AI 모델은?',

'다양한 자연어 처리 및 문서 지능 모델을 활용한 AI Engine은?',

'Chatbot의 다양한 Pain Point 은 어느 회사의 것인가?',

'Chatbot의 다양한 Pain Point 은 제한된 학습 데이터로 인한 오답률 증가 이다',

'Chatbot의 다양한 Pain Point 은 신규 데이터 학습에 따른 유지보수의 어려움 이다',

'Chatbot의 Pain Point은 모든 시나리오 작성에 따른 초기 구축 기간 증가, 제한된 학습 데이터로 인한 오답률 증가, 신규 데이터 학습에 따른 유지보수의 어려움을 포함하는가?',

'Chatbot의 신규 데이터 학습에 따른 유지보수의 어려움을 포함하는 것은?',

'제한된 학습 데이터로 인한 오답률 증가, 신규 데이터 학습에 따른 유지보수의 어려움을 포함하는 것은?',

'App 페이지로부터 사용자의 예상 질문을 자동 생성하여 구축 시간 단축 및 성능 향상 을 가능하게 하는 것은?',

'사용자의 질문 의도와 의미를 파악한 검색으로 답변 정확도 향상이 이루어지는 매체는?',

'A-Call을 통한 구축시간 단축 및 성능 향상은 어느 페이지로부터 사용자의 예상 질문을 자동 생성하는가?',

'A-Call을 통한 구축시간 단축 및 성능 향상은 App 페이지로부터 예상 질문을 자동 생성하여 구축 시간 단축, 사용자의 질문 의도와 의미를 파악한 검색으로 답변 정확도 향상을 포함하는가?',

'구직자와 기업 모두에게 복잡한 절차로 인해 많은 시간이 소요되는 채용 프로세스는?',

'서로에게 적합한 인재와 기업을 찾는 데에 발생하는 여러 가지 문제점은 무엇인가?',

'A-Jobs가 해결하고자 하는 Pain Point은 무엇입니까?',

'AI가 포스트 작성을 도와주어 손쉬운 콘텐츠 생산을 가능하게 한 것은?',

'SuperBlog를 통한 새로운 Blog 서비스 제안은 무엇을 통해 이루어졌는가?',

'TIVINE, WAPL의 다양한 component를 활용한 콘텐츠의 풍부성 제공을 위한 새로운 Blog 서비스 제안은 어디에서 이루어졌는가?'

]

# with open('generated question_samples.tsv', 'r', encoding='utf-8') as fp:
#     text = fp.read().split('\n')
    
# for line in text:
#     sentence = line.split('\t')[0]
#     generated_question_list.append(sentence)
    





# 1번 코퍼스 - 토크나이저 적용 및 인코딩
#####
#corpus = corpus_ban[:10][:]
#corpus = corpus_mun[:30][:]
corpus = generated_question_list[:]

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

corpus = exercise_list[:][:]
# print(exercise_list)
# print(corpus)



###
def predict(corpus, normalize=False):
    ### 파일로 저장 추가
    import csv
    #new_path = "predict/top5_3개씩_to구어체_전체 코퍼스.tsv"
    new_path = "기술토의_style transferred question_변환 결과 데이터.tsv"
    
    with open(new_path, 'w') as new_f:
        tw = csv.writer(new_f, delimiter="\t") # "\t"
        header_lt = ['원래 문장', '마스킹한 문장', '우선 순위', '생성 문장 (변환/증강)', '확률 점수']
        tw.writerow(header_lt)        
        
    
        pip = FillMaskPipeline(model=model, tokenizer=tokenizer, device=0, top_k=5) ### 5로 결정함 !!
        final_list = []
        corpus_size = len(corpus)

        for idx in range(corpus_size):
            try:
                encoded = tokenizer.encode(corpus[idx])
                #print()
                print('*** 원래 문장: ', corpus[idx])
                #print(tokenizer.convert_ids_to_tokens(encoded))
                encoded[-3] = 4 # '[MASK]' 토큰
                encoded[-4] = 4 # '[MASK]' 토큰 (추가됨)
                #print(encoded)
                input_masked_sentence = tokenizer.decode(encoded[1:-1])
                #print('*** 마스킹한 인풋 문장: ', input_masked_sentence)
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
                            #print(elem)
                            #print()

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
                            if '*' in [each_dict['token_str'], elem['token_str']]:
                                continue
                            if '\'' in [each_dict['token_str'], elem['token_str']]:
                                continue               
                            if '&' in [each_dict['token_str'], elem['token_str']]:
                                continue
                            if '!' in [each_dict['token_str'], elem['token_str']]:
                                continue   


                            # 정제
                            nor = seq[:]
                            sep_idx = len(seq)-6 ###
                            seq_1 = seq[sep_idx:][:]


                            if normalize == True:
                                ##### normalize() == True 인지 체크 #####
                #                 from soynlp import *
                #                 from soynlp.normalizer import *
                #                 nor = emoticon_normalize(seq, num_repeats=3)

                                from hanspell import spell_checker
                                output = spell_checker.check(seq_1) ###
                                nor_2 = output.checked

                                #if seq != nor:
                                    #print('정제 전 :', seq)
                                    #print('정제 후 :', nor)
                                    #print() 
                            else:
                                nor_2 = seq_1[:]



                            # 제외
                            bulyong = ['아닙니까', '아십니까', '아닌가요?', '아닌가?', '아닐까요', '아닐까', '어떨까', '어떤가', '‘', '아닌지요', '이니까', '것일까', '것일까요', '일인가', '습니까요?', '것인지', '계십니까', '습니까?', '는요', '잖아요', ' \'', '네요', '데요', '했요', '것이까', '을까', '다요']
                            yes = False

                            for word in bulyong:
                                if word in nor_2:
                                    yes = True
                                    break

                            if yes == True:
                                continue


                            # 저장
                            if seq_1 == nor_2:

                                if nor[-6:] == corpus[idx][-6:]:
                                    continue


                                final_sequence = ''.join(list([seq[:sep_idx], nor_2]))

                                if final_sequence == corpus[idx]:
                                    continue

                                if final_sequence in total_lt.keys():
                                    if total_lt[final_sequence] < (score_1*score_2)*100*100:
                                        total_lt[final_sequence] = (score_1*score_2)*100*100
                                else:
                                    total_lt.setdefault(final_sequence, (score_1*score_2)*100*100)

                            '''        
                            else:
                                print('정제 전 :', seq_1)
                                print('정제 후 :', nor_2)
                                print()  
                            '''    



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


                '''
                print(len(total_lt))
                print(total_lt)
                '''
                total_lt_lt = list([k, e] for (k, e) in total_lt.items())
                total_lt_lt.sort(key=lambda each_set: -each_set[1])

                iii = 0
                for (sentence, score) in total_lt_lt[:3]: ### 변경 가능
                    ###print(sentence, '     (', score, ')    ')
                    iii += 1
                    result = list([corpus[idx], input_masked_sentence, iii, sentence, round(score, 2)])
                    ###final_list.append(result)
                    
                    tw.writerow(result)

            except:
                continue
        
    
    new_f.close()
    
    ###return final_list
        
        
'''        
def save(final_list):        
    ### 파일로 저장 추가
    import csv
    new_path = "predict/top5_3개씩_to구어체.tsv"
    
    with open(new_path, 'w') as new_f:
        tw = csv.writer(new_f, delimiter="\t") # "\t"
        header_lt = ['원래 문장', '마스킹한 문장', '우선 순위', '생성 문장 (변환/증강)', '확률 점수']
        tw.writerow(header_lt)
        
        for each_set in final_list:
            tw.writerow(each_set)
            
    #####new_f.close()
'''        
        
        
# f1_list = predict(exercise_list, normalize=False)
# save(f1_list)

###corpus = exercise_list + corpus_mun

#f2_list = predict(corpus_mun, normalize=True)
f2_list = predict(generated_question_list, normalize=True)
#save(f2_list)
