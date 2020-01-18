from django.db import models
from baseModel.BaseModel import BaseModel
# Create your models here.

class Movie(BaseModel):
    movie_id = models.AutoField(primary_key = True, verbose_name = "电影id")
    movie_name = models.CharField(max_length = 100, unique = True, verbose_name = "电影片名")
    movie_length = models.IntegerField(default = 0, verbose_name = "电影片长")
    movie_releaseTime = models.DateField(verbose_name = "上映时间")
    movie_intro = models.CharField(max_length=512,null=True, verbose_name='剧情简介')
    movie_cover = models.ImageField(upload_to='MovieCover/%Y/%m/%d',null=True, verbose_name='电影封面图片路径')
    #封面的字段类型可考虑使用models.ProcessedImageField来进行处理https://blog.csdn.net/weixin_34314962/article/details/88618031
    movie_grades = models.FloatField(default = 0.0,verbose_name = '豆瓣评分')

    def __str__(self):
        return self.movie_name
