import mlxtend
import numpy as np
from tqdm import tqdm_notebook as tqdm
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import association_rules
from flask import Flask, render_template, redirect, request, url_for

# # *******************************************경로 설정 필요
# # 조병근이 보낸 파일 세개 이용할 것
# # 라이브러리 다음에 바로 읽어줘야 메소드에서 읽어서 실행할 수 있으므로 라이브러리 바로 다음에 읽어와줄것
# SW = pd.read_excel('C:/Users/USer\Desktop/졸업작품/22-1 소웨 전공.xlsx')
# prev = pd.read_excel('C:/Users/USer\Desktop/졸업작품/21-1 소웨 전공.xls')
# past = pd.read_excel('C:/Users/USer\Desktop/졸업작품/20-1 소웨 전공.xlsx')

# with open('C:/Users/USer/Desktop/졸업작품/졸작수강기록.csv', encoding='ISO-8859-1') as myfile:
#     total_lines = sum(1 for line in myfile)

# student_ID = int(input('학번을 입력하세요 : '))
# year = int(input('학년을 입력하세요 : '))
# if year == 3:
#     track = str(input('수강하시는 트랙을 입력해주세요(정보보안, 빅데이터, 센서 중 택1): '))
#     if (track == '센서'):
#         SW = SW[SW.교과명 != '데이터과학']
#         SW = SW[SW.교과명 != '정보보호개론']
#     if (track == '정보보안'):
#         SW = SW[SW.교과명 != '데이터과학']
#         SW = SW[SW.교과명 != '센서와 무선 네트워크']
#     if (track == '빅데이터'):
#         SW = SW[SW.교과명 != '정보보호개론']
#         SW = SW[SW.교과명 != '센서와 무선 네트워크']

# elif year == 4:
#     track = str(input('수강하시는 트랙을 입력해주세요(제네럴, 빅데이터, 센서 중 택1): '))
#     if (track == '센서'):
#         SW = SW[SW.교과명 != '딥러닝']
#         SW = SW[SW.교과명 != '클라우드컴퓨팅시스템']
#     if (track == '제네럴'):
#         SW = SW[SW.교과명 != '임베디드 시스템']
#         SW = SW[SW.교과명 != '딥러닝']
#     if (track == '빅데이터'):
#         SW = SW[SW.교과명 != '클라우드컴퓨팅시스템']
#         SW = SW[SW.교과명 != '임베디드 시스템']

# fav_pro = str(input('가장 선호하는 전공교수님을 입력하세요 : '))
# hate_pro = str(input('가장 선호하지 않는 전공교수님을 입력하세요 : '))


# class MatrixFactorization():
#     def __init__(self, R, k, learning_rate, reg_param, epochs, verbose=False):
#         """
#         :param R: rating matrix
#         :param k: latent parameter
#         :param learning_rate: alpha on weight update
#         :param reg_param: beta on weight update
#         :param epochs: training epochs
#         :param verbose: print status
#         """
#         self._R = R
#         self._num_users, self._num_items = R.shape
#         self._k = k
#         self._learning_rate = learning_rate
#         self._reg_param = reg_param
#         self._epochs = epochs
#         self._verbose = verbose

#     def fit(self):
#         """
#         training Matrix Factorization : Update matrix latent weight and bias
#         :return: training_process
#         """

#         # init latent features
#         self._P = np.random.normal(size=(self._num_users, self._k))
#         self._Q = np.random.normal(size=(self._num_items, self._k))

#         # init biases
#         self._b_P = np.zeros(self._num_users)
#         self._b_Q = np.zeros(self._num_items)
#         self._b = np.mean(self._R[np.where(self._R != 0)])

#         # train while epochs
#         self._training_process = []
#         for epoch in range(self._epochs):
#             # rating이 존재하는 index를 기준으로 training
#             xi, yi = self._R.nonzero()
#             for i, j in zip(xi, yi):
#                 self.gradient_descent(i, j, self._R[i, j])
#             cost = self.cost()
#             self._training_process.append((epoch, cost))

#             # print status
#             if self._verbose == True and ((epoch + 1) % 10 == 0):
#                 print("Iteration: %d ; cost = %.4f" % (epoch + 1, cost))

#     def cost(self):
#         """
#         compute root mean square error
#         :return: rmse cost
#         """

#         # xi, yi: R[xi, yi]는 nonzero인 value를 의미한다.
#         xi, yi = self._R.nonzero()
#         # predicted = self.get_complete_matrix()
#         cost = 0
#         for x, y in zip(xi, yi):
#             cost += pow(self._R[x, y] - self.get_prediction(x, y), 2)
#         return np.sqrt(cost / len(xi))

#     def gradient(self, error, i, j):
#         """
#         gradient of latent feature for GD

#         :param error: rating - prediction error
#         :param i: user index
#         :param j: item index
#         :return: gradient of latent feature tuple
#         """

#         dp = (error * self._Q[j, :]) - (self._reg_param * self._P[i, :])
#         dq = (error * self._P[i, :]) - (self._reg_param * self._Q[j, :])
#         return dp, dq

#     def gradient_descent(self, i, j, rating):
#         """
#         graident descent function

#         :param i: user index of matrix
#         :param j: item index of matrix
#         :param rating: rating of (i,j)
#         """

#         # get error
#         prediction = self.get_prediction(i, j)
#         error = rating - prediction

#         # update biases
#         self._b_P[i] += self._learning_rate * (error - self._reg_param * self._b_P[i])
#         self._b_Q[j] += self._learning_rate * (error - self._reg_param * self._b_Q[j])

#         # update latent feature
#         dp, dq = self.gradient(error, i, j)
#         self._P[i, :] += self._learning_rate * dp
#         self._Q[j, :] += self._learning_rate * dq

#     def get_prediction(self, i, j):
#         """
#         get predicted rating: user_i, item_j
#         :return: prediction of r_ij
#         """
#         return self._b + self._b_P[i] + self._b_Q[j] + self._P[i, :].dot(self._Q[j, :].T)

#     def get_complete_matrix(self):
#         """
#         computer complete matrix PXQ + P.bias + Q.bias + global bias

#         - PXQ 행렬에 b_P[:, np.newaxis]를 더하는 것은 각 열마다 bias를 더해주는 것
#         - b_Q[np.newaxis:, ]를 더하는 것은 각 행마다 bias를 더해주는 것
#         - b를 더하는 것은 각 element마다 bias를 더해주는 것

#         - newaxis: 차원을 추가해줌. 1차원인 Latent들로 2차원의 R에 행/열 단위 연산을 해주기위해 차원을 추가하는 것.

#         :return: complete matrix R^
#         """
#         return self._b + self._b_P[:, np.newaxis] + self._b_Q[np.newaxis:, ] + self._P.dot(self._Q.T)

#     # 과목추천
#     # 데이터프레임 final이 추천하는 과목


# def required_rec():
#     data_df = pd.read_csv('C:/Users/USer/Desktop/졸업작품/졸작수강기록.csv', encoding='ISO-8859-1')
#     if __name__ == "__main__":
#         R = data_df.to_numpy()

#     fav_pro30 = fav_pro + "30"
#     fav_pro20 = fav_pro + "20"
#     hate_pro0 = hate_pro + "0"
#     hate_pro_minus = hate_pro + '-'

#     # matrix Factorization 실행
#     factorizer = MatrixFactorization(R, k=3, learning_rate=0.01, reg_param=0.01, epochs=1000, verbose=False)
#     factorizer.fit()
#     result = factorizer.get_complete_matrix()

#     result_df = pd.DataFrame(result)
#     result_df.columns = ["한명묵", "정용주", "최아영", "민연아", "강상우", "이주형", "노웅기", "최재영", "정옥란", "황효석", "이상웅", "유준", "최재혁",
#                          "조정찬", "김원", "최기호", "정윤현", "민홍", "오영민", "차영운", "김철연", "전영철"]

#     # 교수님 별로 받을 수 있는 점수의 최대점이 30점이 넘지 않도록 조정
#     result_df = (result_df * 15).round(-1).astype(int)

#     # 교수님 별로 점수를 매겨 순위를 매김
#     # 예를 들어, 최재영 교수님이 29점, 최기호 교수님이 20점이면 최재영 교수님이 우선순위로 추천되도록 한다.
#     # 이렇게 분석된 결과가 csv 파일로 저장이 된다.
#     result_df['한명묵'] = result_df['한명묵'].map('한명묵{}'.format)
#     result_df['정용주'] = result_df['정용주'].map('정용주{}'.format)
#     result_df['최아영'] = result_df['최아영'].map('최아영{}'.format)
#     result_df['민연아'] = result_df['민연아'].map('민연아{}'.format)
#     result_df['강상우'] = result_df['강상우'].map('강상우{}'.format)
#     result_df['이주형'] = result_df['이주형'].map('이주형{}'.format)
#     result_df['노웅기'] = result_df['노웅기'].map('노웅기{}'.format)
#     result_df['최재영'] = result_df['최재영'].map('최재영{}'.format)
#     result_df['정옥란'] = result_df['정옥란'].map('정옥란{}'.format)
#     result_df['황효석'] = result_df['황효석'].map('황효석{}'.format)
#     result_df['이상웅'] = result_df['이상웅'].map('이상웅{}'.format)
#     result_df['유준'] = result_df['유준'].map('유 준{}'.format)
#     result_df['최재혁'] = result_df['최재혁'].map('최재혁{}'.format)
#     result_df['조정찬'] = result_df['조정찬'].map('조정찬{}'.format)
#     result_df['김원'] = result_df['김원'].map('김 원{}'.format)
#     result_df['최기호'] = result_df['최기호'].map('최기호{}'.format)
#     result_df['정윤현'] = result_df['정윤현'].map('정윤현{}'.format)
#     result_df['민홍'] = result_df['민홍'].map('민 홍{}'.format)
#     result_df['오영민'] = result_df['오영민'].map('오영민{}'.format)
#     result_df['차영운'] = result_df['차영운'].map('차영운{}'.format)
#     result_df['김철연'] = result_df['김철연'].map('김철연{}'.format)
#     result_df['전영철'] = result_df['전영철'].map('전영철{}'.format)

#     # ************************************************************************ 경로 설정 해줘야함
#     # csv 파일을 생성하고 그 csv를 읽어와야 오류가 안남, 뒤에 헤더나 인덱스 건들일 필요x
#     result_df.to_csv('C:/Users/USer/Desktop/졸업작품/sgd.csv', header=False, index=False, encoding='utf-8-sig')
#     sgd_df = pd.read_csv('C:/Users/USer/Desktop/졸업작품/sgd.csv')
#     sgd_df.columns = ["한명묵", "정용주", "최아영", "민연아", "강상우", "이주형", "노웅기", "최재영", "정옥란", "황효석", "이상웅", "유준", "최재혁", "조정찬",
#                       "김원", "최기호", "정윤현", "민홍", "오영민", "차영운", "김철연", "전영철"]
#     sgd_np = sgd_df.to_numpy()
#     te = TransactionEncoder()
#     te_ary = te.fit(sgd_np).transform(sgd_np)
#     df = pd.DataFrame(te_ary, columns=te.columns_)
#     fp = fpgrowth(df, min_support=0.2, use_colnames=True)
#     result_fp = pd.DataFrame(fp)

#     # ************************************************************************ 경로 설정 해줘야함
#     # csv 파일을 생성하고 그 csv를 읽어와야 오류가 안남, 뒤에 헤더나 인덱스 건들일 필요 x
#     result_fp.to_csv('C:/Users/USer/Desktop/졸업작품/fp.csv', header=False, index=False, encoding='utf-8-sig')
#     relation = pd.read_csv('C:/Users/USer/Desktop/졸업작품/fp.csv')
#     relation.columns = ['support', 'itemsets']

#     # 20점 이상의 점수를 받은 교수님들은 어느정도 학생이 선호한다고 생각
#     condition20 = relation[relation['itemsets'].str.contains(fav_pro20)]
#     condition30 = relation[relation['itemsets'].str.contains(fav_pro30)]
#     fav = pd.concat([condition20, condition30])
#     fav = fav.sort_values(by=['support'], axis=0, ascending=False)
#     SW_year = SW['수강학년'] == year
#     search = SW[SW_year]
#     lecture = search[['교과명', '교수성명', '점수']]

#     # 피하고싶은 교수님을 데이터프레임에서 제거
#     fav = fav[~fav['itemsets'].str.contains(hate_pro, na=False, case=False)]
#     fav = fav[~fav['itemsets'].str.contains('-', na=False, case=False)]

#     # 학과에 안계신 교수님들 데이터프레임에서 제거
#     lecture = lecture[~lecture['교수성명'].str.contains('황효석', na=False, case=False)]
#     lecture = lecture[~lecture['교수성명'].str.contains('민연아', na=False, case=False)]

#     # 앞에서 처리한 데이터들을 깔끔하게 정리해서 교수님 별 순위를 정리하여 recommend에 넣음
#     recommend = []
#     for item in fav['itemsets']:
#         if fav_pro in item:
#             item = item.replace('0', "")
#             item = item.replace('2', "")
#             item = item.replace('3', "")
#             item = item.replace('1', "")
#             item = item.replace('4', "")
#             item = item.replace(fav_pro, "")
#             item = item.replace('frozenset({\'', '')
#             item = item.replace('\', \'\'})', '')
#             item = item.replace('\'})', '')
#             item = item.replace('\'', '')
#             item = item.replace(', ', '')
#             item = item.replace(' ', '')
#             recommend.append(item)

#     count = 0
#     final = pd.DataFrame()
#     recommend = pd.DataFrame(recommend)
#     recommend.drop_duplicates(subset=None, keep='first', inplace=True, ignore_index=False)
#     recommend.dropna(inplace=True)
#     recommend.columns = ['교수성명']
#     copy = lecture.copy()
#     copy.drop_duplicates(['교과명'], keep='first', inplace=True, ignore_index=False)
#     num_prolec = len(copy)

#     # 선호하는 교수님이 있다면 그 교수님의 강의를 무조건 추천
#     final = pd.concat([final, lecture[lecture['교수성명'].str.contains(fav_pro)]])

#     for pro in recommend['교수성명']:
#         for prof in lecture['교수성명']:
#             if pro == prof and count != num_prolec:
#                 final = pd.concat([final, lecture[lecture['교수성명'].str.contains(prof)]])
#                 count = count + 1
#                 continue

#     final.drop_duplicates(['교과명'], keep='first', inplace=True, ignore_index=False)

#     for lec1 in final['교과명']:
#         for lec2 in lecture['교과명']:
#             if lec1 == lec2:
#                 lecture = lecture[~lecture['교과명'].str.contains(lec1, na=False, case=False)]

#     lecture = lecture.sort_values(by=['점수'], axis=0, ascending=False)
#     lecture.drop_duplicates(['교과명'], keep='first', inplace=True, ignore_index=False)
#     final = pd.concat([final, lecture])
#     print('전공 추천 목록입니다.')
#     final.drop(['점수'], axis=1, inplace=True)
#     print(final)

#     # new_lec : 새로 개설된 강의
#     # new_cp : 새로 오신 교수님이 개설한 강의
#     # 새로 오신 교수님, 새로 개설된 강의를 따로 추천


# def new_rec():
#     new = pd.DataFrame()
#     new_lec = pd.DataFrame()
#     new_all = pd.concat([prev, past])
#     prev['ver'] = 'old'
#     SW['ver'] = 'new'
#     id_dropped = set(prev['학수번호']) - set(SW['학수번호'])
#     id_added = set(SW['학수번호']) - set(prev['학수번호'])
#     for lec_num1 in id_added:
#         for lec_num2 in SW['학수번호']:
#             if lec_num1 == lec_num2:
#                 condition = (SW.학수번호 == lec_num2)
#                 new = pd.concat([new, SW[condition]])

#     new = new.drop_duplicates(['교과명', '교수성명'])
#     lec_name = new[new['교과명'] == '졸업작품2'].index
#     new.drop(lec_name, inplace=True)
#     new_cp = new.copy()
#     for prof1 in new['교수성명']:
#         for prof2 in prev['교수성명']:
#             if prof1 == prof2:
#                 new_lec = pd.concat([new_lec, new[new['교수성명'].str.contains(prof2)]])
#                 new_lec = new_lec.drop_duplicates(['교과명', '교수성명'])
#                 new = new[~new['교수성명'].str.contains(prof2, na=False, case=False)]

#     for prof1 in new_cp['교수성명']:
#         for prof2 in past['교수성명']:
#             if prof1 == prof2:
#                 new_cp = new_cp[~new_cp['교수성명'].str.contains(prof2, na=False, case=False)]

#     for prof1 in new_cp['교수성명']:
#         for prof2 in prev['교수성명']:
#             if prof1 == prof2:
#                 new_cp = new_cp[~new_cp['교수성명'].str.contains(prof2, na=False, case=False)]

#     new_lec.drop(['ver'], axis=1, inplace=True)
#     new_lec.drop(['점수'], axis=1, inplace=True)
#     new.drop(['ver'], axis=1, inplace=True)
#     new_cp.drop(['ver'], axis=1, inplace=True)
#     new_cp.drop(['점수'], axis=1, inplace=True)
#     condition = (new_cp.수강학년 == year)
#     condition2 = (new_lec.수강학년 == year)
#     print()
#     print('이번 학기에 수강하실 수 있는 강의 중 새로 오신 교수님께서 맡으신 강의입니다.')
#     if new_cp[condition].empty == True:
#         print('새로 오신 교수님이 맡은 강의가 없습니다.')
#     else:
#         print(new_cp[condition])

#     print()
#     print('이번 학기에 수강하실 수 있는 강의 중 새로 개설된 강의입니다.')
#     if new_lec[condition2].empty == True:
#         print('새로 개설된 강의가 없습니다.')
#     else:
#         print(new_lec[condition2])


# # 과목 추천메소드랑 새로 개설된 강의나 교수님 강의 추천 메소드 실행
# required_rec()
# new_rec()


# ############################### TFIDF ###########################################3
# # -*- coding: utf-8 -*-
# import pandas as pd # 데이터프레임 사용을 위해
# from sklearn.metrics.pairwise import cosine_similarity

# # 키워드 csv파일 가져오기
# csv = pd.read_csv('C:/Users/USer/Desktop/졸업작품/2020_1_전공_키워드.csv', encoding='cp949')

# # 과목명 리스트형태로 가져오기
# course_name_list = csv['교과명']
# course_name_list = course_name_list.values.tolist()
# # print(course_name_list)

# # 키워드 가져오기
# keyword1 = csv['keyword']
# keyword_Word = []
# for k in range(len(csv)):

#     # 키워드가 없는 경우 (강의 개요, 강의 목표 없어서 키워드가 없음)
#     if keyword1[k] is 'x':
#         #print('키워드 없음')
#         continue

#     else:
#         keywordList = list(keyword1[k].split('/'))

#         for i in range(len(keywordList)):
#             # 한 강의의 키워드 끝을 나타내는 '' 가 나오면 다음으로 넘어가기
#             if keywordList[i].split(',')[0] == '':
#                 # print('끝')
#                 continue
#             # 이미 키워드 사전에 있는 키워드라면 건너 뛰기
#             elif keywordList[i].split(',')[0] in keyword_Word:
#                 # print('똑같은거')
#                 continue

#             else:
#                 # print(keywordList[i].split(',')[0])
#                 keyword_Word.append(keywordList[i].split(',')[0])

#     #print(keywordList)
# #print()
# #print(keyword_Word)

# # 데이터프레임 만들기
# dataFrame = pd.DataFrame(index=course_name_list, columns=keyword_Word)

# for k in range(len(csv)):
#     course_name = csv.iat[k, 2]
#     #print(course_name)

#     # 키워드가 없는 경우 (강의 개요, 강의 목표 없어서 키워드가 없음)
#     if csv.iat[k, 10] is 'X':
#         #print('키워드 없음')
#         continue

#     elif csv.iat[k, 10] is not 'X':
#         keywordList = list(keyword1[k].split('/'))
#         for j in range(len(keywordList) - 1):
#             #print(keywordList[j].split(', '))
#             #print(keywordList[j].split(', ')[0])  # 키워드 한글
#  
#            #print(keywordList[j].split(', ')[1])  # 숫자
#             dataFrame.loc[course_name, keywordList[j].split(', ')[0]] = keywordList[j].split(', ')[1]

# #display(dataFrame)

# # 비어있는 칸 0으로 채우기
# dataFrame.fillna(0, inplace=True)
# # 데이터프레임 csv 파일로 저장
# dataFrame.to_csv('TFIDF_전공.csv', index=True, encoding='euc-kr')

# # 데이터프레임 어레이로 바꾸기
# # 이거 필요없나봐ㅋㅎ
# # dataFrame_array = dataFrame.to_numpy()

# # 코사인 유사도 계산하기
# cosine_matrix = cosine_similarity(dataFrame, dataFrame)
# #print(cosine_matrix)
# #print()

# # 코사인 유사도 계산한거 csv 파일로 저장하기 (dataframe으로 다시 바꿔서 csv로 저장)
# cosine_matrix_df = pd.DataFrame(cosine_matrix)
# cosine_matrix_df.to_csv('cosine_전공.csv', index=True, encoding='euc-kr')


# course_to_index = dict(zip(csv['교과명'], csv.index))
# # print(course_to_index)

# def get_recommendations(course, cosine_matrix=cosine_matrix):
#     # 사용자가 입력한? 강의 이름의 인덱스를 받아온다.
#     idx = course_to_index[course]

#     # 해당 강의와 모든 강의와의 유사도를 가져온다.
#     sim_scores = list(enumerate(cosine_matrix[idx]))

#     # 유사도에 따라 강의들을 정렬한다.
#     sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

#     # 가장 유사한 10개의 강의를 받아온다.
#     sim_scores = sim_scores[1:11]

#     # 가장 유사한 10개의 강의의 인덱스를 얻는다.
#     movie_indices = [idx[0] for idx in sim_scores]

#     # 가장 유사한 10개의 강의의 제목을 리턴한다.
#     return csv['교과명'].iloc[movie_indices]

# # 사용자가 입력하는걸로 바꾸기
# print()
# print()
# print()
# print("수강했던 과목과 비슷한 과목을 추천해드립니다.")
# course_name = input("수강했던 과목을 입력하세요: ")
# print()
# print(get_recommendations(course_name))
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def main():
    if request.method == 'POST':
        return redirect(url_for('test'))
    return render_template('test.html')

@app.route('/hell', methods=['GET', 'POST'])
def test():
    return render_template('hell.html')

if __name__ == '__main__':
    app.run()

