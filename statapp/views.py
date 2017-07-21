# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.shortcuts import render,render_to_response
from django.http import HttpResponse
import pandas as pd

def helloworld(request):
    return HttpResponse('Hello World')
# Create your views here.
def home(request):
    pd.set_option('display.max_columns', 50)
    pd.set_option('display.width', 200)
    if 'd' in request.GET:
        df = pd.read_csv(request.GET['d'], encoding='gbk').head(5)
    else:
        df=pd.DataFrame()
    col=pd.DataFrame(df.columns).T
    html=render_to_response('home.html',{'df':df,'col':col})
    return HttpResponse(html)

def data(request):
    pd.set_option('display.max_columns',50)
    pd.set_option('display.width', 8000)
    df=pd.read_csv(request.GET['d'],encoding='gbk')
    df1=df.iloc[:,0]
    col=pd.Series(df.columns)
    return render_to_response('data.html',{'df':df,'col':col,'df1':df1})

