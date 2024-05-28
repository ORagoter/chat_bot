import psycopg2

dbname='postgres' #имя базы данных
user='postgres.zzyahwklsrihlglqbsfd' # имя пользователя
password='xy9$G/Cy~b~)&+_' # пароль
host='aws-0-eu-central-1.pooler.supabase.com'  # хост
port='5432' # порт

conn = psycopg2.connect(
        dbname='postgres', #имя базы данных
        user='postgres.zzyahwklsrihlglqbsfd', # имя пользователя
        password='xy9$G/Cy~b~)&+_', # пароль
        host='aws-0-eu-central-1.pooler.supabase.com',  # хост
        port='5432' # порт
    )