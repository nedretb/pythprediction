import mysql.connector
from datetime import datetime, timedelta

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  database="test"
)

mycursor = mydb.cursor()

#mycursor.execute("SELECT * FROM test")

#myresult = mycursor.fetchall()


lastFiveDays = [1179, 1265, 1135, 703, 1555]
mid = 0.99
low = 0.98
high = 1.25

date = datetime.date(datetime.now())
counter = 1
for x in lastFiveDays:
    sql = "INSERT INTO `future`(`date`, `low`, `mid`, `high`) VALUES (%s, %s, %s, %s)"
    val = (date + timedelta(days=counter), x*low, x*mid, x*high)
    mycursor.execute(sql, val)

    mydb.commit()
    print(date + timedelta(days=counter))
    #print(x*low)
    #print(x*mid)
    #print(x*high)
    counter = counter +1

#for x in myresult:
 # print(x)