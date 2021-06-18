FROM python:3.7.5

WORKDIR /app

COPY requirements.txt /app
COPY /naive_deep_Q_learning /app

RUN pip3 install -r requirements.txt

CMD python cartpolegame.py