import mysql.connector

mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="ektlsThr10075%",
    database="ai_in_the_wild"
)

mycursor = mydb.cursor()
mycursor.execute("CREATE DATABASE ai_in_the_wild")
mycursor.execute("SHOW DATABASES")
for x in mycursor:
    print(x)

