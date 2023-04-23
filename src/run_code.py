import sqlite3


def run_sql(database_path, query):
    # '/home/simin/Project/KnowPrompt/Dataset/spider/database/yelp/yelp.sqlite'
    # open the database file
    conn = sqlite3.connect(database_path)

    # create a cursor object
    cursor = conn.cursor()

    # execute a query to retrieve data from the database
    cursor.execute(query)

    # retrieve the data from the query
    rows = cursor.fetchall()

    # close the database file
    conn.close()
    return rows
