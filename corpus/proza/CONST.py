import datetime

BASE_URL = "https://proza.ru"
TEXTS_URL = BASE_URL + "/texts"

topic_num = str
END_DATE = datetime.datetime(year=2022, month=1, day=1)


class TOPIC:
    FANTASY: topic_num = "24"
    NOVEL: topic_num = "04"
