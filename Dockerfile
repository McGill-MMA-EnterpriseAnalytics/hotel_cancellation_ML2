FROM python:3.11

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

#
COPY ./preprocessing.py /code/
COPY ./bestModel_hpo.pkl /code/ 
COPY ./bestModel_tpot.pkl /code/ 
COPY ./fastapi_app.py /code/

# 
CMD ["uvicorn", "fastapi_app:app", "--host", "0.0.0.0", "--port", "80"]