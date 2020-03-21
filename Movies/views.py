from django.shortcuts import render,redirect
from django.http import HttpResponse
from Movies.models import Movie,MovieLab
from Comments.models import Comments
from Filmmakers.models import Celebrity
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
    comment_objects = Comments.objects.filter(movie=movie)
    comment_list = []
    if comment_objects:
        for comment in comment_objects:
            comment_list.append({"user": comment.user.user_name,"content":comment.content})

    lab_objects = movie.lab.all()  # return all labs objects for this movie
    lab_list = []
    if lab_objects:
        for lab in lab_objects:
            lab_list.append({"type": lab.lab_content, "url": "#"})

    type_objects = movie.types.all()  # return all type objects for this movie
    types = ''
    if type_objects:
        for movie_type in type_objects:
            types = types + movie_type.type_name + ','
        types = types[:-1]

    director_objects = Celebrity.objects.filter(movie=movie, roletable__role='director')
    directors = ''
    for director in director_objects:
        directors = directors + director.celebrity_name + ' / '
    directors = directors[:-3]

    actor_objects = Celebrity.objects.filter(movie=movie, roletable__role='actor')
    actors = ''
    actor_imgs = []
    if actor_objects:
        for actor in actor_objects:
            actors = actors + actor.celebrity_name + ' '
            actor_imgs.append({"img": "/image/"+str(actor.celebrity_cover)})
        actors = actors[:-1]

    result = {
        # "id": movie_id,
        "title": movie.movie_name,
        "name": movie.movie_name,
        "poster": "/image/" + str(movie.movie_cover),
        "showtime": movie.movie_releaseTime.strftime('%Y-%m-%d'),
        "showpos": movie.movie_showPos,
        "length": movie.movie_length,
        "type": types,
        "director": directors,
        "actor": actors,
        "score": movie.movie_grades,
        "introduction": movie.movie_intro,
        "actorimg": actor_imgs,
        "lab": lab_list,
        "comment": comment_list,

    }
    return HttpResponse(json.dumps(result,ensure_ascii=False), content_type="application/json", charset='utf-8')


def get_movies(request):
    type = request.GET.get('type')  # 获取请求的类别值
    print(type)

    result = {
        "data": [
            {
                "title": "千与千寻",
                "score": "9.3",
                "date": "2019",
                "type": "剧情,动画,奇幻",
                "img": "/image/MovieCover/20200131/p2557573348.webp",
                "url": "../detail/1"
            }
        ]
    }
    return HttpResponse(json.dumps(result,ensure_ascii=False), content_type="application/json", charset='utf-8')
