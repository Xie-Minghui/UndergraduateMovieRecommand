from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from Users.models import User


@csrf_exempt
def login(request):
    usr = request.GET.get('usr')
    pwd = request.GET.get('pwd')
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
