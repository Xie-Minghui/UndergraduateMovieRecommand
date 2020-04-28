import json

from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from Users.models import User
from Recommend.LFM_sql import LFM, ReadMysql
from Recommend.LFM_test import lfm, sparse_matrix

@csrf_exempt
def login(request):
    if request.method == 'POST':
        params = json.loads(request.body)
        print(params)
        usr = params['params']['usr']
        pwd = params['params']['pwd']
        print(usr+'=='+pwd)
        response = HttpResponse()
        try:
            usrObj = User.objects.get(user_name=usr)
            if usrObj.password == pwd:
                request.session['user'] = usr
                response['result'] = 'true'
                print('登陆成功')
            else:
                response['result'] = 'false'
                print('密码错误')
        except:
            response['result'] = 'false'
            print('用户未找到')
        return response


Configuration = {
    'host': "localhost",
    'username': "root",
    'password': "112803",
    'database': "mrtest"
}


def getRecommendMovies(request, userID):

    RecommendMovies = lfm.RecommendtoUser(userID, 4, sparse_matrix)
    # print(RecommendMovies)
    return HttpResponse(RecommendMovies)
