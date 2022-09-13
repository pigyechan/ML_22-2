import pandas as pd #판다스 라이브러리 호출
import numpy as np #넘파이 라이브러리 호출
import random #난수 라이브러리 호출
import matplotlib.pyplot as plt #시각화를 위한 matplotlib 라이브러리 호출

def distance(x, y): #distance 함수를 만들어 필요할 때 사용할 수 있도록 함.
    return int(np.sqrt(pow((x[0]-y[0]),2)+pow((x[1]-y[1]),2)))

lis=[]
lisr=[] #난수로 저장된 리스트 값들을 튜플로 저장하기 위한 두 개의 리스트를 생성

for i in range(10): 
    #for문을 이용해서 0~20 사이의 난수로 이루어진 리스트 10개를 저장
    lis.append([random.randint(0,20), random.randint(0,20)])

lise = pd.DataFrame(lis,columns=['x','y'])
plt.scatter(lise['x'], lise['y'])
plt.xlabel('x')
plt.ylabel('y') #lis를 데이터프레임 형태로 변수 lise에 저장해서 시각화하기 위함

for i in range(10):
    #위에서 만들어진 리스트를 튜플로 저장하고 다시 새로운 리스트 안에 저장
    a = tuple(lis[i])
    lisr.append(a)
    

def knn(x, y, k):
    #knn 함수로 거리를 도출하고 거리에 따라 오름차순으로 정렬한 다음 k값만큼 출력
    result=[]
    for i in range(10):
        result.append([distance(x,y[i]), y[i]]) #result 리스트에 거리와 그에 따른 점을 차례로 저장
    result_1 = sorted(result, 
                key=lambda result: result[0], reverse = False) #result에 저장된 점들의 거리가 가까운 순대로 정렬.
    print(result_1) #어떤 점들이 얼마만큼의 거리에 떨어져 있는지 오름차순으로 정렬된 리스트 출력.
    result_2 =[]
    result_3 =[] #다시 한번 데이터 프레임을 만들기위한 새로운 리스트
    for i in range(k):
        result_2.append(result_1[i]) #입력받은 k의 갯수만큼 result_2 리스트에 저장.
        result_3.append(result_2[i][1]) #result_3 리스트는 데이터 프레임을 만들기 위한 용도.
    print(result_2)
    resul = pd.DataFrame(result_3,columns=['x','y']) 
    plt.scatter(resul['x'], resul['y']) 
    #새로운 데이터 프레임을 생성하고 입력한 점과 가장 가까운 점 표시.

(a,b)= input("숫자 두 개 입력? ").split(",")
new = int(a), int(b)
#문자로 숫자 두개를 입력받고 int형으로 변환

num = int(input("k를 입력해주세요: ")) 
#k를 입력받고 바로 int형으로 변환하여 몇 개를 출력할 건지 변수 num에 저장

knn(new, lisr, num) #knn 함수를 통해 가장 가까운 거리에 있는 점들을 k의 갯수만큼 거리까지 구함.

plt.scatter(new[0],new[1]) #입력한 값도 시각화.
