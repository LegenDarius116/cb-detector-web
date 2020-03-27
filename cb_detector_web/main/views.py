from django.shortcuts import render
from .forms import TextForm

from joblib import load


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

            context['bully_prediction'] = bool_result
            context['bully_probability'] = "{0:.2f}".format(probability_result[1]*100)

    context['form'] = TextForm()
    return render(request, 'main/index.html', context)
