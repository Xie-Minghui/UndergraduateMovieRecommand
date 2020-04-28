"""UndergraduateMovieRecommand URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path,include
from Movies import views as M_v
from Users import views as U_v


urlpatterns = [
    path('user/', include('Users.urls')),
    path('control/', admin.site.urls),  # 修改默认后台入口路径为control/
    path('movies/',include('Movies.urls')),
    # path('api/home_movies.json',M_v.return_home_movies),
    path('api/getMovies/',M_v.get_movies),
    # path('api/getRecommendMovies'),
    path('api/getMovieDetail/<int:movie_id>',M_v.return_movie_json),
    path('api/login',U_v.login),
]
