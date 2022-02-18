from __future__ import print_function

import pandas as pd
from flask import Flask, jsonify, request
import json
import MF as mf
import content_based as cb
import pymysql
import MySQLdb
import mysql.connector
import csv

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
    respons = []  # chứa những gì đã được sửa
    get_all_rows_from_table_statistic_mysql = []
    # Thực hiện thuật toán retrain lại từ đầu
    # Lấy tất cả dữ liệu từ bảng statistic (#lấy ra các thông tin: id bài đăng (item), id_user (người dùng), score_rated, times_visitted)
    for id_statistic, id_news, id_user, score_rated, times_visitted in cur.fetchall():
        get_all_rows_from_table_statistic_mysql.append([id_statistic, id_news, id_user, score_rated, times_visitted])

    for each_of_rate in get_all_rows_from_table_statistic_mysql: # duyệt các row trong bảng statistic (mysql)
        # cần each_of_rate[1] và each_of_rate[2] thôi
        all_scores = each_of_rate[3] if each_of_rate[3] is not None else 0
        all_times = each_of_rate[4] if each_of_rate[4] is not None else 0
        r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
        df = pd.read_csv('ml-100k/ub.base', sep='\t', names=r_cols, encoding='latin-1')

        # tìm trong csv
        row = -1
        with open('ml-100k/ub.base') as o:
            myData = csv.reader(o, delimiter='\t')
            for r in myData:
                if (int(r[0]) == int(each_of_rate[2])) & (int(r[1]) == int(each_of_rate[1])):
                    row = myData.line_num - 1  # đã biết cần phải sửa row nào rồi!
                    respons.append({row:r}.copy())   #các dòng cũ chuẩn bị xóa
        # sửa trong ml-100k/ub.base nếu và chỉ nếu tìm được vị trí để sửa
        if row != -1:
            visiting_score = 0
            if 0 <= all_times <= 1:
                visiting_score = 2
            elif 1 < all_times <= 3:
                visiting_score = 5
            elif 3 < all_times <= 5:
                visiting_score = 8
            elif all_times > 5:
                visiting_score = 10
            score = (2 * int(df.loc[row, 'rating']) + all_scores + visiting_score) / 3
            score = int(score / 2)
            if score > 5:
                score = 5
            elif score < 0:
                score = 0
            df.loc[row, 'rating'] = score
        df.to_csv("ml-100k/ub.base", header=None, sep='\t', encoding='latin-1', index=False)
        # sửa xong rồi thì xóa toàn bộ bảng statistic trong mysql
    r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
    df = pd.read_csv('ml-100k/ub.base', sep='\t', names=r_cols, encoding='latin-1')
    print("--------------ĐÃ TRAIN XONG")
    cur.execute("TRUNCATE TABLE statistic")

    # retrain lại mô hình 1 lần nữa:
    rate_train = df.values
    rate_train[:, :2] -= 1
    rs = mf.MF(Y_data=rate_train, K=10, lam=.1, print_every=10,
               learning_rate=0.75, max_iter=100, user_based=1)
    rs.fit()
    rs.write_X_W_to_csv()
    # Thực hiên xong thuật toán retrain lại từ đầu
    myConnection.close()
    return jsonify(respons)


def get_old_id_user():
    id_users = []
    for i in range(len(dict(request.args.lists())) - 1):
        id_users.append(int(dict(request.args.lists())['id_news[' + str(i) + ']'][0]))
    return id_users


if __name__ == '__main__':
    app.run(debug=True, host='192.168.0.107', port=8000)
