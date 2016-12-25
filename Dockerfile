FROM amitkgupta/python-2.7-machine-learning

MAINTAINER Diep Dao <diepdao12892@gmail.com>

RUN apt-get update \
    && apt-get install -y libxml2-dev libxslt1-dev zlib1g-dev libjpeg62-turbo-dev

ENV PYTHONUNBUFFERED 1
RUN mkdir /code
WORKDIR /code

RUN pip install -U pip && pip install -U flask flask-restplus tornado gunicorn

RUN pip install -U pattern gensim==0.13.3 theano tensorflow tensorflow-gpu keras
