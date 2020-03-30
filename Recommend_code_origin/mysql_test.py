import pymysql

def test_connection(host,username,password,database):
    db = pymysql.connect(host,username,password,database)
    cursor = db.cursor()
    
    try:
        cursor.execute("select ratingID from ratings where ratingID <= '%d'" % (100))
        results = cursor.fetchall()
        print(results)
        ratingID = []
        for item in results:
            ratingID.append(item[0])
        print(ratingID)
        # db.commit()
    except:
        print("Error")
    db.close()

def SelectData(host,username,password,database):
    db = pymysql.connect(host,username,password,database)
    cursor = db.cursor()
    cursor.execute("select userID, movieID, rating from ratings")
    results = cursor.fetchall()
    userID, movieID, rating = [],[],[]
    for item in results:
        userID.append(item[0])
        movieID.append(item[1])
        rating.append(item[2])
    print('userID:')
    print(userID)
    print('movieID:')
    print(movieID)
    print('rating:')
    print(rating)
    
    db.close()



if __name__ == "__main__":
    host = "localhost"
    username = "root"
    password = "112803"
    database = "mrtest"
    # test_connection(host,username,password,database)
    SelectData(host, username, password,database)