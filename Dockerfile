FROM jupyter/scipy-notebook

WORKDIR /code
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

CMD jupyter notebook --ip 0.0.0.0 --allow-root --no-browser --port 5000
