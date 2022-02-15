from __future__ import print_function

import pandas as pd
from flask import Flask, jsonify, request
import json
import MF as mf
import content_based as cb
import pymysql
import MySQLdb
import mysql.connector

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False


# y = json.dumps(x)
# news = request.get_json()


@app.route('/evaluate', methods=['GET'])
def evaluate():
    id_news = get_old_id_user()
    id_user = request.args.get('id_user')
    u = int(id_user)

    print("--------------------------------Những gì đã gửi lên: ", id_news)

    X_new = pd.read_csv('X_MF.csv', sep=',', header=None).values
    W_new = pd.read_csv('W_MF.csv', sep=',', header=None).values
    B_new = pd.read_csv('B_MF.csv', sep=',', header=None).values

    score = {}

    for each_id_news in id_news:
        i = int(each_id_news)
        bias = B_new[u]
        pred = X_new[i, :].dot(W_new[:, u]) + bias  # pred = X
        if pred < 0:
            score[each_id_news] = 0
        elif pred > 5:
            score[each_id_news] = 5
        else:
            score[each_id_news] = pred[0]

    score_for_each_id_news = {k: v for k, v in sorted(score.items(), key=lambda item: item[1])}

    print("--------------------------------Sau khi tính toán: ", score_for_each_id_news)

    # mảng id_news phải được sắp xếp lại số sao dựa theo id_user
    return jsonify(score_for_each_id_news, id_user)


@app.route('/retrain', methods=['GET'])
def retrain():
    hostname = 'localhost'
    username = 'root'
    password = 'a123456A'
    database = 'imarket'

    # chạy MySQL
    myConnection = MySQLdb.connect(host=hostname, user=username, passwd=password, db=database)
    # myConnection = mysql.connector.connect(host=hostname, user=username, passwd=password, db=database)
    # myConnection = pymysql.connect(host=hostname, user=username, passwd=password, db=database)
    cur = myConnection.cursor()
    cur.execute("SELECT * FROM statistic")
    respon = []
    for id_statistic, id_news, id_user, score_rated, times_visitted in cur.fetchall():
        respon.append([id_statistic, id_news, id_user, score_rated, times_visitted])
    myConnection.close()
    return jsonify(respon)


def get_old_id_user():
    id_users = []
    for i in range(len(dict(request.args.lists())) - 1):
        id_users.append(int(dict(request.args.lists())['id_news[' + str(i) + ']'][0]))
    return id_users


if __name__ == '__main__':
    app.run(debug=True, host='192.168.0.107', port=8000)
