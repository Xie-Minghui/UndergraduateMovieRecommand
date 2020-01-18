from django.shortcuts import render,redirect
from django.http import HttpResponse
from Movies.models import Movie
import os
import json
#from django.core.serializers.json import json

def test(request):
    # obj = Movie.objects.get(movie_id = 1)
    # m = dict()
    # m['movie_name'] = obj.movie_name
    # m['cover'] = obj.movie_cover
    # #path = os.path.join()
    # print(obj.movie_cover)
    return render(request,'react/index.html')

def Return_Home_movies(request):
    #print('======')
    result = {"success": True,
              "data":{
                  "movies":[
                      {"id": 1,
                       "title": "铤而走险",
                       "img": "https://img1.doubanio.com/view/photo/s_ratio_poster/public/p2566515717.webp",
                       "url": "/movie/1"
                       },
                      {"id": 2,
                       "title":"美丽人生",
                       "img": "https://img3.doubanio.com/view/photo/s_ratio_poster/public/p2578474613.webp",
                       "url": "movie/2"
                      }
                  ]}}
    return HttpResponse(json.dumps(result,ensure_ascii=False),content_type = "application/json",charset='utf-8')