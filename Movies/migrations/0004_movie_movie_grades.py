# Generated by Django 3.0.2 on 2020-01-18 12:59

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('Movies', '0003_auto_20191004_1442'),
    ]

    operations = [
        migrations.AddField(
            model_name='movie',
            name='movie_grades',
            field=models.FloatField(default=0.0, verbose_name='豆瓣评分'),
        ),
    ]