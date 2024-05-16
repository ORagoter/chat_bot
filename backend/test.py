import config
import psycopg2
import os

print(config.dbname)
# Установка параметров подключения к базе данных Supabase
conn_params = {
    "database": os.environ.get(config.dbname),
    "user": os.environ.get(config.user),
    "password": os.environ.get(config.password),
    "host": os.environ.get(config.host),
    "port": os.environ.get(config.port)
}

# Создание соединения с базой данных
conn = psycopg2.connect(**conn_params)

# Создаем объект cursor для выполнения SQL-запросов
cur = conn.cursor()

# Выполняем SQL-запрос для извлечения данных из таблицы processed_data
cur.execute("SELECT name_of_block, question, answer FROM processed_data")

# Получаем все строки результата запроса
rows = cur.fetchall()

# Закрываем соединение и курсор
cur.close()
conn.close()
# Печатаем первые несколько строк, чтобы убедиться, что данные были успешно загружены
for row in rows[:5]:
    print(row)