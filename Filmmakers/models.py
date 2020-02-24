from django.db import models
from baseModel.BaseModel import BaseModel
# Create your models here.
class Celebrity(BaseModel):#明星信息表
    genderType = (('male', "男"),('female', "女"),)#定义性别选项

    celebrity_id = models.AutoField(primary_key = True, verbose_name = "明星id")
    celebrity_name = models.CharField(max_length = 20, unique = True, verbose_name = "明星名字")
    celebrity_gender = models.CharField(max_length=10,choices=genderType,default="男", verbose_name='明星性别',null=True)
    celebrity_birthyear = models.CharField(max_length=4,verbose_name='出生日期',null=True)
    celebrity_intro = models.TextField(max_length=500,verbose_name='明星简介',null=True)