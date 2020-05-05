
import pickle
# from sklearn.externals import joblib
import joblib
# from Recommend.LFM_sql import LFM, ReadMysql
from django.conf import settings
from Recommend.KNN41 import KNN, Data_process
Configuration = {
    'host': settings.DATABASES['default']['HOST'],
    'port': settings.DATABASES['default']['PORT'],
    'username': settings.DATABASES['default']['USER'],
    'password': settings.DATABASES['default']['PASSWORD'],
    'database': settings.DATABASES['default']['NAME']
}
# lfm = LFM(lfm_num=10)  # lfm_num 设置模型隐向量的维度
knn = KNN()
origin_data = Data_process(
    Configuration['host'], Configuration['port'], Configuration['username'], Configuration['password'], Configuration['database'])
#   print("begin")
try:
    # with open('knn.pkl','rb') as f:
        # knn = pickle.load(f.read())
    # knn = joblib.load(r'E:\MR\UndergraduateMovieRecommand\Recommend\knn0.m')
    with open(r'E:\MR\UndergraduateMovieRecommand\Recommend\knn0.m', 'rb') as f:
        knn = joblib.load(f)
    # print(knn.similarity_item_matrix[:10,:60])
except IOError:
    print("KNN File not exist!")
# try:
#     with open(r'E:\MyProject_test\Recommend_code_origin\lfm_sql.pkl', 'rb') as f:  # E:\MyProject_test\Recommend_code_origin\
#         lfm = pickle.loads(f.read())
# except IOError:
#     print("File not exist!")
# sparse_matrix = ReadMysql(
#     Configuration['host'], Configuration['username'], Configuration['password'], Configuration['database'])
