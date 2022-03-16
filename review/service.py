from bs4 import BeautifulSoup
from urllib.request import Request, urlopen
import joblib, re
from konlpy.tag import Twitter
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

class ModelService:
    
    def __init__(self):
        self.vec = None # 벡터 객체화
        self.twitter = Twitter() # 한글 토큰화
    
    # train, test 파일 읽어서 데이터 프레임으로 변환 함수
    def read_dataFile(self, path):
        df = pd.read_csv(path, sep='\t')
        df = df.fillna(' ')
        X = df['document'].apply(lambda x: re.sub(r"\d+", " ", x))
        y = df['label']
        return X, y
    
    def tw_tokenizer(self, text):
        # 입력 인자로 들어온 text 를 형태소 단어로 토큰화 하여 list 객체 반환
        tokens_ko = self.twitter.morphs(text)
        return tokens_ko
    
    def tran_fit(self, fit_data):
        self.vec = TfidfVectorizer(tokenizer=self.tw_tokenizer, ngram_range=(1,2), min_df=3, max_df=0.9)
        self.vec.fit(fit_data)
    
    def data_vectorizer(self, data):
        data_vec = self.vec.transform(data)
        return data_vec
        
    def fit(self, X, y):
        lr = LogisticRegression(random_state=0)
        lr.fit(X, y)
        return lr
    
    def test(self, model, X, y):
        pred = model.predict(X)
        return accuracy_score(y, pred)
    
    def saveFile(self, model, fname):
        joblib.dump(model, fname)
        
    def loadFile(self, fname):
        return joblib.load(fname)
    
    
class ReviewService: # 학습, 평가
     
    def __init__(self):
        self.model = None # 파일에서 로드한 모델 저장할 변수
        self.modelservice = ModelService() 
        
    def review_fit(self):
        # 학습 파일 로드 및 전처리
        X_train, y_train = self.modelservice.read_dataFile('static/ratings_train.txt')
        # 학습 데이터를 벡터 변환기에 설정=> 사전 생성
        self.modelservice.tran_fit(X_train)
        # 학습 데이터를 사전 기반으로 벡터화 작업
        X_vec = self.modelservice.data_vectorizer(X_train)
        # 학습
        self.model = self.modelservice.fit(X_vec, y_train)
        # 학습 모델을 파일로 저장
        self.modelservice.saveFile(self.model, 'static/movie_review_type.pkl')
        
    def review_test(self):
        # 테스팅 데이터 파일 로드 및 전처리
        X_test, y_test = self.modelservice.read_dataFile('static/ratings_test.txt')
        # 테스팅 데이터 벡터화
        X_vec = self.modelservice.data_vectorizer(X_test)
        # test()로 평가
        score = self.modelservice.test(self.model, X_vec, y_test)
        print('score:', score)
        return score
    
    def review_pred(self, df, model_path):
        # 멤버 변수 모델이 널이면 테스팅할 모델이 없으므로
        if self.model == None:
            # 이미 학습한 파일 모델을 로드
            self.model = joblib.load(model_path)
            # 학습 데이터를 로드해서 사전생성
            X_train, y_train = self.modelservice.read_dataFile('static/ratings_train.txt')
            self.modelservice.tran_fit(X_train)

        # 예측할 실제 데이터 로드 및 전처리
        X_data = df['review'].apply(lambda x: re.sub(r"\d+", " ", x))

        # 학습 때 생성한 사전으로 벡터화 작업
        X_vec = self.modelservice.data_vectorizer(X_data)
        print(X_vec[:10])
        # 벡터화된 데이터로 예측
        pred = self.model.predict(X_vec)
        df['pred'] = pred
        
        return df
       
    def type_change(self, pred):
        return 'positive' if pred == 1 else 'negative'
        
    def getMovieReviews(self, sword):       
        date_list = []
        writer_list = [] 
        review_list = []
        rating_list = []
        
        for page in range(1, 5):
            url = 'https://movie.naver.com/movie/point/af/list.naver?st=mcode&target=after'
            url += '&sword=' + str(sword) 
            url += '&page=' + str(page)
            
            print('>> ' + str(page))     
                   
            try: 
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.116 Safari/537.36'}
                request = Request(url, headers=headers) 
                response = urlopen(request)
                html = response.read()
                soup = BeautifulSoup(html, 'lxml')
                reviews = soup.select('#old_content > table > tbody > tr') 

                for r in reviews:
                    date = r.select_one('td:nth-child(3)').text
                    writer = r.select_one('td:nth-child(3) > a').text
                    review = r.find('td', attrs={'class': 'title'}).text
                    rating = r.find('em').text
                    
                    date = date.split('****')[1].strip()
                    review = review.split('\n')[5].strip()
                    print(date, writer, review, rating)
                    
                    if review != '':
                        date_list.append(date)
                        writer_list.append(writer)
                        review_list.append(review)
                        rating_list.append(rating)

            except Exception as e:
                pass
        
        data = pd.DataFrame(list(zip(date_list, writer_list, review_list, rating_list)), 
                columns = ['date', 'writer', 'review', 'rating'])

        df = self.review_pred(data, 'static/movie_review_type.pkl')  
        df['pred_type'] = df['pred'].apply(self.type_change) 
        
        return df

    
if __name__ == "__main__": 
    s = ReviewService()
    res = s.getMovieReviews(189559) 