from tensorflow import keras
import random
import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

# 데이터 로드
data = pd.read_csv('신체검사데이터.csv', encoding='cp949')

# 결측치 확인
# print(data.isnull().sum())

# 순번            0
# 측정 일자         0
# 가슴 둘레 센티미터    0
# 소매길이 센티미터     0
# 신장 센티미터       0
# 허리 둘레 센티미터    0
# 샅높이 센티미터      0
# 머리 둘레 센티미터    0
# 발 길이 센티미터     0
# 몸무게 킬로그램      0
# dtype: int64

# 데이터 타입 검사
# print(data.dtypes)

# 순번              int64
# 측정 일자           int64
# 가슴 둘레 센티미터    float64
# 소매길이 센티미터     float64
# 신장 센티미터       float64
# 허리 둘레 센티미터     object   -> 문자열 데이터 : 의심스러움
# 샅높이 센티미터      float64
# 머리 둘레 센티미터    float64
# 발 길이 센티미터     float64
# 몸무게 킬로그램      float64
# dtype: object

# 비수치값 샘플보기
# 숫자가 아닌 값을 포함하는 행 필터링
non_numeric_data = data[data['허리 둘레 센티미터'].str.contains(r'[^\d\.]', regex=True, na=False)]
# r'[^\d\.]' 정규표현식 패턴, \d : 숫자, \. : 소수점 의미, [^...] : 괄호안에 있는 문자를 제외한 모든 문자와 일치 
# 즉, 숫자(0-9)와 소수점(.)을 제외한 모든 문자 찾는 것
# regex=True: 문자열 검색을 위해 정규 표현식을 사용하겠다는 옵션
# na=False: 결측치(NaN)는 검사에서 제외하겠다는 의미

# 샘플 출력
print(non_numeric_data['허리 둘레 센티미터'])