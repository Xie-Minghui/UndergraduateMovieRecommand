
import pickle
from Recommend_code_origin.LFM_sql import LFM, ReadMysql
Configuration = {
    'host': "localhost",
    'username': "root",
    'password': "112803",
    'database': "mrtest"
}
lfm = LFM(lfm_num=10)  # lfm_num 设置模型隐向量的维度
try:
    with open(r'E:\MyProject_test\Recommend_code_origin\lfm_sql.pkl', 'rb') as f:
        lfm = pickle.loads(f.read())
except IOError:
    print("File not exist!")
sparse_matrix = ReadMysql(
    Configuration['host'], Configuration['username'], Configuration['password'], Configuration['database'])
