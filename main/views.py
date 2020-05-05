from django.shortcuts import render
from .forms import TextForm

from joblib import load


def index(request):
    context = {}

    if request.method == 'GET':
        context['form'] = TextForm()
    elif request.method == 'POST':
        form = TextForm(request.POST)
        context['form'] = form

        if form.is_valid():
            # get the content of entered text
            body = form.cleaned_data['text']

            # load the machine learning model
            text_clf = load('main/model/model_v3.model')

            # predict if text is bullying or not (binary)
            bool_result = int(text_clf.predict([body])[0]) == 1

            # show confidence that the prediction is indeed cyberbullying
            probability_result = text_clf.predict_proba([body])[0][1]*100

            # rounding probability result to 2 decimal places (in percent form)
            probability_result = round(100*probability_result) / 100

            # print logs
            print(f"Bullying Prediction: {bool_result}")
            print(f"Prediction Confidence: {probability_result}%")

            # pick color of alert for toxicity rating (confidence)
            if probability_result < 30:
                toxicity_rate_alert_color = "alert alert-success"
            elif 30 <= probability_result < 50:
                toxicity_rate_alert_color = "alert alert-warning"
            else:
                toxicity_rate_alert_color = "alert alert-danger"

            # pick color of alert for binary prediction
            bullying_prediction_alert_color = "alert alert-success" if not bool_result else "alert alert-danger"

            # add these to context object
            context['toxicity_rate_alert_color'] = toxicity_rate_alert_color
            context['bully_prediction'] = bool_result
            context['bully_probability'] = probability_result
            context['bullying_prediction_alert_color'] = bullying_prediction_alert_color

    return render(request, 'main/index.html', context)
