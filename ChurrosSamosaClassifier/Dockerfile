FROM python:3.9

COPY requirements.txt .

RUN pip install --upgrade -r requirements.txt

COPY server server/

EXPOSE 5000

CMD ["sh", "-c", "python server/main.py && sleep infinity"]
