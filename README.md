# Padded MLM for Style Transfer

Padded MLM(Masked Language Model)을 활용한 문체 변환

<br>

## Task
Target 문체의 데이터셋으로 학습시킨 모델(MLM)로 Source 문체의 데이터셋을 변환/증강


#### 1. 문어체 → 구어체 변환
<img width="1259" alt="스크린샷 2022-06-24 오후 7 06 40" src="https://user-images.githubusercontent.com/70500841/175520078-a7717b8e-ca79-4931-bb60-025df7ee70f3.png">

#### 2. 구어체 → 문어체 변환
<img width="1259" alt="스크린샷 2022-06-24 오후 7 06 32" src="https://user-images.githubusercontent.com/70500841/175520146-b5e711f7-22da-46e3-a5ed-ebb3354d11c4.png">


<br>

## installation
#### 1. 환경 설정
```
conda install python==3.8
conda install tqdm
conda install transformers
conda install progressbar2
conda install -c conda-forge cupy
conda install jupyter notebook
conda install ipykernel
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

#### 2. 정제 기법
- **soynlp**
```
$ pip install soynlp
```

```
$ git clone https://github.com/lovit/soynlp.git
$ python setup.py install
```


- **py-hanspell**
```
$ git clone https://github.com/ssut/py-hanspell.git
$ python setup.py install
```


<br>

## Dataset
#### 1. 데이터셋 출처(**Source Dataset**)

: KoQuAD 1.0, AI Hub - 일반상식, AI Hub - 기계독해, AI Hub - 도서 기계독해

<br>

#### 2. `0_style data_extractor_json to txt.py` 파일

: Source Dataset(Json)에서 'question' 추출할 때 **3가지** 어체로 "**문체 분리**" 적용함

<img width="1252" alt="스크린샷 2022-06-24 오후 7 10 30" src="https://user-images.githubusercontent.com/70500841/175520198-93466759-fc40-492b-893d-9327412688c8.png">


이 중 `1. 문어체` 와 `3. 반말체` 데이터셋을 사용

- 선정 Source Dataset

<img width="1261" alt="스크린샷 2022-06-24 오후 7 18 23" src="https://user-images.githubusercontent.com/70500841/175520223-6bad4847-b1ba-443a-a95e-ed5338fcfbf3.png">


<br>

#### 3. `0_make_dataset_preprocessing_and_split.py` 파일


: 데이터 전처리(정제) 후, 전체 코퍼스 파일(`total corpus.tsv`)과 이를 8 : 1 : 1 비율로 split한 `train.tsv` , `dev.tsv` , `test.tsv` 파일로 저장하는 코드



<br>

## Model Training
(`2_padded_MLM-finetuning-ban.ipynb` 파일)

#### 1. 토크나이저 및 모델
- 사용한 **Tokenizer** 및 **ModelForMaskedLM** : `klue/bert-base`
```
from transformers import AutoModel, AutoTokenizer, AutoModelForMaskedLM, AdamW

tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
model = AutoModelForMaskedLM.from_pretrained("klue/bert-base")
```
- 해당 Tokenizer의 **vocab 사전** (3만 2천 개) : `vocab.txt` 파일 참고

<br>

#### 2. 모델 훈련 방식
- 임베딩 변경

  : 3가지 대안으로 실험해봄

     1) 기본 MLM
     2) padded MLM (선정)
     3) segment id까지 변화시켜 본 padded MLM
  
  
- Tokenizer & DataLoader 과정

<img width="1260" alt="스크린샷 2022-06-24 오후 6 51 55" src="https://user-images.githubusercontent.com/70500841/175520783-8a71f66f-247b-477e-9ee0-cdb141acdfd3.png">



<br>

#### 3. 학습된 모델 결과물
- 1 . 문어체 → 구어체 변환 : model/ 의 `padded_MLM_train_ban_4_epoch` 파일

- 2 . 구어체 → 문어체 변환 : model/ 의 `padded_MLM_train_mun_4_epoch` 파일



<br>

## Model Inference
(`4_padded_MLM-inference-make_final result dataset.py` 파일)

#### 1. Masking 과정

: Tokenizer 함수 사용 (convert_ids_to_tokens 함수 → [MASK] 토큰으로 교체 → decode 함수)


#### 2. [MASK] 예측

: FillMaskPipeline 라이브러리 사용

- 적절한 **[MASK] 개수** 선정

  : 3가지 대안으로 실험해봄
     1) [MASK] 1개
     2) [MASK] 2개 (선정)
     3) [MASK] 3개




<br>


## Reference
- 마스크 언어 모델 기반 비병렬 한국어 텍스트 스타일 변환 (2021, NCSOFT NLP Center Language AI Lab)
- Unsupervised Text Style Transfer with Padded Masked Language Models (2020, Google Research)

<br>
