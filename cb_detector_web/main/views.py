from django.shortcuts import render
from .forms import TextForm

from joblib import load

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def index(request):
    context = {}

    if request.method == 'POST':
        form = TextForm(request.POST)
        if form.is_valid():
            # get the content of entered text
            body = form.cleaned_data['text']

            # load the machine learning model
            text_clf = load('main/model/model_v1.model')

            bool_result = text_clf.predict([body])[0] == 1
            probability_result = text_clf.predict_proba([body])[0]

            print(probability_result)

    context['form'] = TextForm()
    return render(request, 'main/index.html', context)
