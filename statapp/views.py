# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.shortcuts import render,render_to_response
from django.http import HttpResponse
import pandas as pd

def helloworld(request):
    return HttpResponse('Hello World')
# Create your views here.
def home(request):
    html=render_to_response('home.html')
    return HttpResponse(html)

def data(request):
    pd.set_option('display.max_columns',50)
    pd.set_option('display.width', 8000)
    df=pd.read_csv(request.GET['d'],encoding='gbk')
    df5=df.head(10)
    col=pd.Series(df.columns)
    return render_to_response('data.html',{'df':df,'col':col})

