from django.shortcuts import render,redirect
from django.http import HttpResponse
from Movies.models import Movie
import os
import json
#from django.core.serializers.json import json


def return_home_movies(request):
    #print('======')
    # result = {"success": True,
    #           "data":{
    #               "movies":[
    #                   {"id": 1,
    #                    "title": "铤而走险",
    #                    #"img": "https://img1.doubanio.com/view/photo/s_ratio_poster/public/p2566515717.webp",
    #                    "img": "/upload/MovieCover/2020/01/31/p2557573348.webp",
    #                    "url": "/movie/1"
    #                    },
    #                   {"id": 2,
    #                    "title":"美丽人生",
    #                    "img": "https://img3.doubanio.com/view/photo/s_ratio_poster/public/p2578474613.webp",
    #                    "url": "movie/2"
    #                   }
    #               ]}}

    #按照上映时间抽取最新条目
    movie_objects = Movie.objects.all().order_by('-movie_releaseTime')[:2]
    id_num = 1
    movies = []
    for movie in movie_objects:
        print(movie.movie_name)
        movies.append({
            "id": id_num,
            "title": movie.movie_name,
            "img": "/image/" + str(movie.movie_cover),
            "url": "/movie/" + str(movie.movie_id)
        })
        id_num += 1

    result = {
        "success": True,
        "data": {
            "movies": movies
        }
    }
    
    return HttpResponse(json.dumps(result,ensure_ascii=False), content_type="application/json", charset='utf-8')

def return_movie_json(request, movie_id):
    movie = Movie.objects.get(movie_id=movie_id)

    result = {
        "data": {},
        "message": {},
        "剧情简介": "",
        "评分": "",
        "评价人数": ""
    }
    return HttpResponse(json.dumps(result,ensure_ascii=False), content_type="application/json", charset='utf-8')