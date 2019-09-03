from django.db import models
from baseModel.BaseModel import BaseModel
# Create your models here.
class UsersManager(models.Manager): #用户管理函数
	def add_one_account(self,user_name,password,image_path):
		user = self.create(user_name = user_name,password = gethash(password),image_path = image_path)
		return user
	def delete_one_account(self,user_name,password):
		try:
			user = self.get(user_name = user_name,password = password)
		except self.model.DoesnotExist
			user = None
		return user
class Users(BaseModel):
	user_id = models.AutoField(primary_key = True,verbose_name = "用户id") #自增类型
    user_name = models.CharField(max_length = 20,unique = True,verbose_name = "用户名")
    password = models.CharField(max_length = 20,verbose_name = "用户密码")
    signature = models.CharField(max_length = 20,verbose_name = "个性签名")
    email = models.EmailField(verbose_name='用户邮箱')
    is_actived = models.BooleanField(default=False, verbose_name='激活状态')
    image_path = models.CharField(max_length = 50,verbose_name = "头像图片路径")
    gender = models.CharField(max_length=10,default="male", verbose_name='性别')

    browser_list = models.CharField(max_length=300,default="[]",verbose_name='浏览')#浏览表的id
    browser_number = models.IntegerField(default=0,verbose_name='浏览量')
    have_watched = models.CharField(max_length=300,default="[]",verbose_name='观看')#已看过表的id
    watch_number = models.IntegerField(default=0,verbose_name='观看量')
    collection_list = models.CharField(max_length=300,default="[]",verbose_name='收藏')#收藏表的id
    collect_number = models.IntegerField(default=0,verbose_name='收藏量')

    prefer_actors = models.CharField(max_length=300,default="[]",verbose_name='偏爱演员')#偏爱演员表的id
    prefer_directors = models.CharField(max_length=300,default="[]",verbose_name='偏爱的导演')#偏爱导演表的id

    recommend_movies = models.CharField(max_length=300,default="[]",verbose_name='推荐的电影列表')#推荐的电影列表的id

    objects = UsersManager()#用户管理类

