from django.db import models
from baseModel.BaseModel import BaseModel
# Create your models here.

class Movie(BaseModel):
    movie_id = models.AutoField(primary_key = True, verbose_name = "电影id")
    movie_name = models.CharField(max_length = 100, unique = True, verbose_name = "电影片名")
    movie_length = models.IntegerField(default = 0, verbose_name = "电影片长")
    movie_releaseTime = models.DateField(verbose_name = "上映时间")

    def __str__(self):
        return self.movie_name
