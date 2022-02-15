import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import Ridge

# Reading user file:
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('ml-100k/u.user', sep='|', names=u_cols,
                    encoding='latin-1')

n_users = users.shape[0]  # chỉ quan tâm id của user thôi, = 943
'''
users.head() = 
    user_id  age sex  occupation zip_code
0        1   24   M  technician    85711
1        2   53   F       other    94043
2        3   23   M      writer    32067
3        4   24   M  technician    43537
4        5   33   F       other    15213
'''
# Reading ratings file:
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']

ratings_base = pd.read_csv('ml-100k/ua.base', sep='\t', names=r_cols, encoding='latin-1')
ratings_test = pd.read_csv('ml-100k/ua.test', sep='\t', names=r_cols, encoding='latin-1')

rate_train = ratings_base.values
rate_test = ratings_test.values

'''
Number of traing rates: rate_train = 90570
Number of test rates: rate_test = 9430
'''

# Reading items file:
i_cols = ['movie id', 'movie title', 'release date', 'video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
          'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
          'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

items = pd.read_csv('ml-100k/u.item', sep='|', names=i_cols, encoding='latin-1')

n_items = items.shape[0]
'''
Number of items: n_items =  1682
'''
X0 = items.values
X_train_counts = X0[:, -19:]
'''
X_train_counts là 1 ma trận cỡ (1682, 19) gồm toàn số 0 và 1
X_train_counts lấy 19 cột bên phải của ma trận u_item: 1682 items đó thuộc những thể loại nào trong 19 thể loại được liệt kê ở i_cols
'''
# tfidf chính là việc chuẩn hoaas ma trận X_train_counts, hay ma trận X trong công thức

transformer = TfidfTransformer(smooth_idf=True, norm='l2')
tfidf = transformer.fit_transform(X_train_counts.tolist()).toarray()
'''
tfidf = 
[[0.         0.         0.         ... 0.         0.         0.        ]
 [0.         0.53676706 0.65097024 ... 0.53676706 0.         0.        ]
 [0.         0.         0.         ... 1.         0.         0.        ]
 ...
 [0.         0.         0.         ... 0.         0.         0.        ]
 [0.         0.         0.         ... 0.         0.         0.        ]
 [0.         0.         0.         ... 0.         0.         0.        ]]
là 1 ma trận cỡ (1682, 19) = (M * K), có vai trò như ma trận X trong công thức (nhưng đã được chuẩn hóa)
'''


def get_items_rated_by_user(rate_matrix, user_id):
    """
    in each line of rate_matrix, we have infor: user_id, item_id, rating (scores), time_stamp
    we care about the first three values
    return (item_ids, scores) rated by user user_id
    """
    y = rate_matrix[:, 0]  # all users
    # item indices rated by user_id
    # we need to +1 to user_id since in the rate_matrix, id starts from 1
    # while index in python starts from 0
    ids = np.where(y == user_id + 1)[0]
    item_ids = rate_matrix[ids, 1] - 1  # index starts from 0
    scores = rate_matrix[ids, 2]
    '''
    y           :cột 0 trong ma trận ua.base
    ids         :các dòng mà có sự đánh giá của user_id trong ma trận ua.base
    item_ids    :các item mà user_id đánh giá
    scores      :số sao mà user_id dánh giá
    VD:
    y =  [  1   1   1 ... 943 943 943] (90570,)
    ids =  [32776 32777 32778 32779 32780 32781 32782 32783 32784 32785 32786 32787
     32788 32789 32790 32791 32792 32793 32794 32795 32796 32797 32798 32799
     32800 32801 32802 32803 32804 32805 32806 32807 32808 32809 32810 32811
     32812 32813 32814 32815 32816 32817 32818 32819 32820 32821 32822 32823
     32824 32825 32826 32827 32828 32829 32830 32831 32832 32833 32834 32835] (60,)
    item_ids =  [   0    7   10   21   30   31   57   58   63   80  123  131  159  174
      177  179  181  197  213  214  220  222  233  237  241  267  268  276
      285  301  303  304  305  413  466  474  481  490  502  505  510  513
      633  652  681  693  701  704  734  810  867  932  946  957 1016 1100
     1140 1193 1198 1295] (60,)
    scores =  [1 3 2 4 2 4 3 5 4 5 4 3 5 4 3 5 4 4 3 3 4 4 4 4 4 5 5 4 4 5 5 5 5 4 3 3 2
     3 4 2 5 3 3 3 5 4 3 2 4 4 4 3 5 5 2 4 3 3 1 5] (60,)
    '''
    # lấy các item đã được đánh giá bởi user_id trong rate_matrix
    return (item_ids, scores)   #trả về (item và điểm số cho item đó)


d = tfidf.shape[1]  # data dimension
W = np.zeros((d, n_users))
b = np.zeros((1, n_users))

'''
d = 19 là 19 thể loại phim = dimensions, là số K trong công thức
b = bias, 1 vector M chiều gồm toàn các số giống nhau (bn)
W = tập tham số, 1 ma trận cỡ K * N cột
'''
#duyệt tất cả người dùng
for n in range(n_users):    #n = 1 đến N = 943
    ids, scores = get_items_rated_by_user(rate_train, n)
    clf = Ridge(alpha=0.01, fit_intercept=True)
    Xhat = tfidf[ids, :]

    clf.fit(Xhat, scores)
    W[:, n] = clf.coef_
    b[0, n] = clf.intercept_
    '''
    ids:    các items mà người dùng n đánh giá
    scores: điểm mà người dùng n đánh giá cho các items đó
    Xhat:   Ma trận X xóa đi các hàng (tương ứng các items) mà người dùng n không đánh giá
    W: Ma trận W trong công thức, cỡ K * N = 19 * 943, ở đây chỉ cập nhật cột thứ n, tức W[:, n]
    b: vector có N = 943 số, ban đầu gán tất cả = 0, tại vòng lặp thứ n thì vị trí n được cập nhật
    
    VD: n = 9
    ids = [  0   3   8  10  11  12  21  22  31  32  39  47  49  55  58  59  63  68
          69  81  84  92  97  98 115 123 126 128 131 132 133 134 136 143 152 154
         155 156 159 160 161 163 167 169 173 175 177 178 179 181 182 184 185 190
         191 193 194 196 197 198 199 202 204 210 215 217 220 222 229 233 237 244
         268 272 273 274 275 282 285 288 293 301 318 320 332 333 339 356 366 370
         384 403 413 417 419 429 431 434 446 461 462 466 469 473 474 477 478 479
         481 482 483 488 492 494 495 496 497 498 501 504 508 509 510 512 517 518
         520 524 526 528 529 530 557 581 587 588 601 602 603 605 609 614 616 628
         650 651 653 654 655 656 662 663 685 691 692 693 694 695 696 697 698 699
         700 701 702 703 704 705 706 707 708 709 710 711] (174,)
        scores = [4 4 4 4 5 3 5 5 4 4 4 4 5 5 4 3 4 4 4 4 4 4 4 5 4 5 5 4 5 5 5 5 4 4 4 4 4
         5 4 4 4 4 4 4 4 4 5 5 5 5 5 5 4 5 4 4 4 5 3 4 5 4 5 5 4 4 4 5 4 4 4 4 4 4
         4 4 4 4 4 4 3 4 3 4 4 4 4 5 4 4 4 4 4 4 4 3 4 5 4 3 4 4 4 4 4 5 5 5 4 5 5
         4 4 4 5 4 5 4 4 4 4 5 4 4 4 5 4 5 4 3 4 5 4 4 4 5 5 5 4 5 4 4 5 4 4 3 5 5
         5 4 3 4 4 4 4 5 3 4 3 4 4 4 4 3 5 3 4 4 5 4 4 4 4 4] (174,)
        Xhat [[0.         0.         0.         ... 0.         0.         0.        ]
         [0.         0.71065158 0.         ... 0.         0.         0.        ]
         [0.         0.         0.         ... 0.         0.         0.        ]
         ...
         [0.         0.         0.         ... 0.         0.         0.        ]
         [0.         0.         0.         ... 0.         0.         0.        ]
         [0.         0.         0.         ... 0.         0.         0.        ]] (174, 19)
        W[:, n] = [ 0.         -0.03256292  0.38539743 -0.77138129  1.38868476 -0.13602652
          0.49892291  0.11147267  0.28633026 -0.04150965  0.46896473  0.57871839
          0.12311276  0.62368815  0.17251723  0.40216874  0.13846745  0.47242557
          0.723668  ] (19, 943)
        b = [[2.99318944 0.1126961  1.35394381 6.00754844 2.00817146 2.48968922
          3.38031137 3.39877507 6.70841735 3.88815576 ...(toàn 0)...]] (1, 943)
    '''

# predicted scores
Yhat = tfidf.dot(W) + b #y = x*w + b

n = 10  #xét người dùng thứ 11
np.set_printoptions(precision=2)  # 2 digits after .
ids, scores = get_items_rated_by_user(rate_test, n)
'''
Yhat:  [[2.92 4.   1.9  ... 5.04 4.78 3.17]
 [2.8  3.44 3.3  ... 1.1  4.76 3.7 ]
 [3.48 1.56 1.27 ... 7.36 4.19 3.81]
 ...
 [4.13 4.1  3.02 ... 9.54 4.64 3.27]
 [3.57 3.45 2.93 ... 3.68 3.66 2.76]
 [4.13 3.8  3.1  ... 9.54 4.42 3.93]] (1682, 943) => M * N = 1682 * 943
ids : [ 37 109 110 226 424 557 722 724 731 739]
scores     : [3 3 4 3 4 3 5 3 3 4]
Yhat[ids, n]: [3.18 3.13 3.42 3.09 3.35 5.2  4.01 3.35 3.42 3.72]
'''


def evaluate(Yhat, rates, W, b):
    se = 0
    cnt = 0
    for n in range(n_users):    #duyệt tất cả người dùng
        ids, scores_truth = get_items_rated_by_user(rates, n)   #lấy số sao tại tập test, trừ đi số sao tính được
        scores_pred = Yhat[ids, n]
        e = scores_truth - scores_pred  #y - y mũ
        se += (e * e).sum(axis=0)
        cnt += e.size
    return np.sqrt(se / cnt)


print('RMSE for training:', evaluate(Yhat, rate_train, W, b))
print('RMSE for test    :', evaluate(Yhat, rate_test, W, b))

'''
K = 19 = số thể loại phim
N = 943 = số users
M = 1628 = số items
'''