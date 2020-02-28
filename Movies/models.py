from django.db import models
from baseModel.BaseModel import BaseModel
# Create your models here.

class Movie(BaseModel):
    movie_id = models.AutoField(primary_key = True, verbose_name = "电影id")
    movie_name = models.CharField(max_length = 100, unique = True, verbose_name = "电影片名")
    movie_length = models.IntegerField(default = 0, verbose_name = "电影片长")
    movie_releaseTime = models.DateField(verbose_name = "上映时间")
    movie_showPos = models.CharField(max_length=10,default='中国大陆',verbose_name="上映地区")
    movie_intro = models.CharField(max_length=512,null=True, verbose_name='剧情简介')
    movie_cover = models.ImageField(upload_to='MovieCover/%Y%m%d',null=True, verbose_name='电影封面图片路径')
    # 封面的字段类型可考虑使用models.ProcessedImageField来进行处理https://blog.csdn.net/weixin_34314962/article/details/88618031
    movie_grades = models.FloatField(default = 0.0,verbose_name = '豆瓣评分')
    # movie_ratingNum = models.IntegerField(default=0,verbose_name='评价人数')
    # movie_origin = models.CharField(max_length=10,default='中国大陆',verbose_name='制片国家/地区')
    # movie_language = models.CharField(max_length=50,default='汉语普通话 / 英语',verbose_name='语言')
    movie_alias = models.CharField(max_length=100,null=True,verbose_name='电影别名')

    types = models.ManyToManyField(to='MovieType')  # 电影-类别 多对多关系表
    lab = models.ManyToManyField(to='MovieLab')  # 电影-标签 多对多关系表
    roles = models.ManyToManyField('Filmmakers.Celebrity', through='RoleTable')  # 电影的导演及演员表

    def __str__(self):
        return self.movie_name


class RoleTable(models.Model): # 角色表，存放电影的演员及导演信息
    roleType = (('director', '导演'), ('actor', '演员'))  # 定义角色选项

    celebrity = models.ForeignKey('Filmmakers.Celebrity', on_delete=models.CASCADE)  # 明星
    movie = models.ForeignKey(Movie, on_delete=models.CASCADE)  # 电影
    role = models.CharField(max_length=15, choices=roleType, verbose_name='在电影中的角色')


class MovieType(models.Model):  # 电影类别表
    type_id = models.AutoField(primary_key=True, verbose_name='电影类别id')
    type_name = models.CharField(max_length=10, unique=True, verbose_name='电影类别名称')

    def __str__(self):
        return self.type_name


class MovieLab(models.Model): # 电影标签
    lab_id = models.AutoField(primary_key=True,verbose_name="电影标签id")
    lab_content = models.CharField(max_length=10,verbose_name="电影标签内容")

    def __str__(self):
        return self.lab_content
