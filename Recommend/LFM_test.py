
import pickle
from Recommend.LFM_sql import LFM, ReadMysql
from django.conf import settings
Configuration = {
    'host': settings.DATABASES['default']['HOST'],
    'port': settings.DATABASES['default']['PORT'],
    'username': settings.DATABASES['default']['USER'],
    'password': settings.DATABASES['default']['PASSWORD'],
    'database': settings.DATABASES['default']['NAME']
}
lfm = LFM(lfm_num=10)  # lfm_num 设置模型隐向量的维度
try:
    with open(r'E:\MR\UndergraduateMovieRecommand\Recommend\lfm_sql.pkl', 'rb') as f:
        lfm = pickle.loads(f.read())
except IOError:
    print("File not exist!")
sparse_matrix = ReadMysql(
    Configuration['host'], Configuration['port'], Configuration['username'], Configuration['password'], Configuration['database'])
