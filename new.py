import mysql.connector
from datetime import datetime, timedelta

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  database="test"
)

sql= 'SELECT `date` FROM (SELECT `id`, `date` FROM `future` ORDER BY id DESC LIMIT 1) sub ORDER BY id ASC'

mycursor = mydb.cursor()

mycursor.execute(sql)

result = mycursor.fetchall()
print(result[0][0])

date = datetime.date(datetime.now())
dateFuture = date + timedelta(days=5)
print(dateFuture)

if result[0][0] == dateFuture:
    print('same')


