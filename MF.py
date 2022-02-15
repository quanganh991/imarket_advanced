import numpy as np
import pandas as pd


class MF(object):
    """docstring for CF"""

    # Khởi tạo, constructor, methods, attributes,...
    def __init__(self, Y_data, K=10, lam=0.1, Xinit=None, Winit=None,
                 learning_rate=0.5, max_iter=1000, print_every=100, user_based=1):
        self.Y_raw_data = Y_data  # rate_train
        self.K = K
        # regularization parameter
        self.lam = lam
        # learning rate for gradient descent
        self.learning_rate = learning_rate
        # maximum number of iterations
        self.max_iter = max_iter
        # print results after print_every iterations
        self.print_every = print_every
        # user-based or item-based
        self.user_based = user_based
        # number of users, items, and ratings. Remember to add 1 since id starts from 0
        self.n_users = int(np.max(Y_data[:, 0])) + 1
        self.n_items = int(np.max(Y_data[:, 1])) + 1
        self.n_ratings = Y_data.shape[0]

        '''
        N = n_users =  943
        M = n_items =  1682
        s = n_ratings =  90570
        K =  10
        '''

        if Xinit is None:  # new
            self.X = np.random.randn(self.n_items, K)
        else:  # or from saved data
            self.X = Xinit

        if Winit is None:
            self.W = np.random.randn(K, self.n_users)

        else:  # from daved data
            self.W = Winit

            """
            Xinit chưa được khởi tạo, lấy ngẫu nhiên 
            X =  [[ 1.52671899  0.61159492 -0.25588742 ... -0.35077568  0.24221038  1.151814  ]
                 [-0.13447599  0.92060397 -0.58548826 ... -0.66634176  0.04266447
                   0.63825065]
                 [ 0.03711261  1.76948442 -1.70657784 ...  0.07057532  1.01635719
                  -1.86545668]
                 ...
                 [ 1.30886301  0.88557013  0.18498807 ... -0.70522761 -0.21136191
                   1.09166551]
                 [-0.66641371  0.29122056 -1.25149684 ...  0.96229084 -0.14443621
                   0.40428259]
                 [-0.37726776  0.12635235 -1.57297613 ... -0.02701525 -1.26199573
                   0.37923836]] là 1 ma trận cỡ M * K = 1682 x 10
                   
            Winit chưa được khởi tạo, lấy ngẫu nhiên
            
            W =  [[-0.50296678 -0.41272054 -1.79649436 ... -0.14860998  0.38282652  1.21059139]
                 [ 0.17597047 -1.32965983  2.20133404 ... -0.37323984 -1.18196545
                  -2.15646584]
                 [-0.2236704   0.18987694 -0.48435053 ...  1.46080738 -0.17715202
                  -1.58085599]
                 ...
                 [-1.72124215 -0.06241452 -2.43341036 ... -0.6790702  -0.27052504
                   2.24360759]
                 [-0.70887354 -0.94360816 -1.03278484 ... -1.0183332   0.61737002
                   1.38795186]
                 [ 0.33258694 -0.46801669  0.02768115 ...  0.19181486 -0.35334321
                   0.75017317]] là 1 ma trận cỡ K * N = 10 x 943

            """

        # normalized data, update later in normalized_Y function
        self.Y_data_n = self.Y_raw_data.copy()
        '''
        Copy nông, chỉ thay đổi dữ liệu trên Y_data_n chứ không thay đổi dữ liệu trên Y_raw_data
        '''

    def normalize_Y(self):
        if self.user_based:  # nếu chọn chế độ lọc cộng tác dựa trên người dùng (mặc định 1)
            user_col = 0  # thì cột 0 là cột người dùng
            item_col = 1  # cột 1 là cột item
            n_objects = self.n_users  # số lượng người dùng = N

        # if we want to normalize based on item, just switch first two columns of data
        else:  # item bas   #nếu chọn chế độ lọc cộng tác dựa trên sản phẩm (user_based = 0)
            user_col = 1  # thì cột 0 là cột người dùng
            item_col = 0  # cột 1 là cột item
            n_objects = self.n_items  # số lượng items = M

        '''
        users: Lấy cột 0 hoặc cột 1 (cột user hoặc item trong ma trận rate_train tùy thuộc chế độ user_based)
        
        trong tình huốn này:
        users =  [  0   0   0 ... 942 942 942] (90570) = cột người dùng
        mu: ban đầu là 1 vector có n_objects số 0 = n_items hoặc n_users số 0 = 1682 hoặc 943 số 0
        '''

        users = self.Y_raw_data[:, user_col]
        self.mu = np.zeros((n_objects,))

        for n in range(n_objects):  # n hoặc = M, hoặc = N :  duyệt số người dùng hoặc số sản phẩm
            # mỗi vòng lặp ko duyệt từng dòng 1, mà duyệt tất cả các dòng của cùng 1 user
            # trong tình huống mặc định: user_based = 1 thì là duyệt tất cả người dùng
            ids = np.where(users == n)[0].astype(
                np.int32)  # chạy từ 0 đến 90569: đánh giá thứ ids trong tất cả những đánh giá
            item_ids = self.Y_data_n[ids, item_col]  # item được đánh giá
            # Y_data_n ban đầu là rate_train
            ratings = self.Y_data_n[ids, 2]  # số sao dành cho item nằm ở dòng ids
            m = np.mean(ratings)
            '''
            ids: dòng thứ bao nhiêu trong file ma trận rate_train tính bắt đầu từ 0
            item_ids: cột item trừ đi 1 = các item mà 1 người dùn
            rating: cột rating = tất cả các rating của 1 người dùng nào đó
            m:  trung bình rating của 1 người dùng nào đó
            VD: n = 0:
            ids =  [  0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17
              18  19  20  21  22  23  24  25  26  27  28  29  30  31  32  33  34  35
              36  37  38  39  40  41  42  43  44  45  46  47  48  49  50  51  52  53
              54  55  56  57  58  59  60  61  62  63  64  65  66  67  68  69  70  71
              72  73  74  75  76  77  78  79  80  81  82  83  84  85  86  87  88  89
              90  91  92  93  94  95  96  97  98  99 100 101 102 103 104 105 106 107
             108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125
             126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143
             144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161
             162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179
             180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197
             198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215
             216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233
             234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251
             252 253 254 255 256 257 258 259 260 261]
            item_ids =  [  0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  17  18
              19  20  21  22  23  24  25  26  27  28  29  30  31  32  33  34  35  36
              37  38  39  40  41  42  43  44  45  47  48  49  50  51  52  53  54  55
              56  57  58  59  60  61  62  64  65  66  67  68  69  70  71  72  73  74
              75  76  77  78  79  80  81  82  83  84  85  86  87  88  90  92  93  94
              95  96  97  98  99 100 101 102 103 104 105 106 107 108 109 110 111 113
             114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131
             132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149
             150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 165 166 167
             168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185
             186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203
             204 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219 220 222
             223 224 225 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242
             243 244 245 246 247 248 249 250 251 253 254 255 256 257 258 259 260 261
             262 263 264 265 266 267 268 269 270 271]
            ratings =  [5 3 4 3 3 5 4 1 5 3 2 5 5 5 5 5 4 5 4 1 4 4 3 4 3 2 4 1 3 3 5 4 2 1 2 2 3
             4 3 2 5 4 5 5 4 5 3 5 4 4 3 3 5 4 5 4 5 5 4 3 2 4 4 3 4 3 3 3 4 3 1 4 4 4
             1 4 4 5 5 3 4 3 5 5 4 5 5 5 2 4 5 3 4 3 5 2 2 1 1 2 4 4 5 5 1 5 1 5 5 3 3
             3 5 1 4 3 4 5 3 2 5 4 5 3 1 4 4 4 4 3 5 1 3 1 3 2 1 4 2 4 3 2 2 5 4 5 3 5
             2 4 4 3 3 4 4 4 4 3 5 5 2 5 5 5 5 5 5 5 5 5 5 5 3 3 5 4 5 4 4 4 4 3 3 5 5
             4 4 4 5 5 5 5 4 3 3 5 4 5 3 4 5 5 4 4 3 4 2 4 3 5 3 3 1 3 5 5 5 2 3 4 4 1
             3 2 4 5 4 2 4 4 3 4 5 1 2 2 5 1 4 4 4 4 2 1 2 4 4 5 1 1 1 3 1 2 4 1 4 5 5
             5 2 3]
            m =  3.5877862595419847
            
            n=1:
            ids =  [262 263 264 265 266 267 268 269 270 271 272 273 274 275 276 277 278 279
             280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297
             298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313]
            item_ids =  [  0   9  12  13  18  24  49  99 110 126 236 241 250 254 257 268 271 272
             273 274 275 276 277 279 280 281 282 283 284 285 286 287 288 289 290 291
             292 293 294 295 296 297 299 301 303 304 305 308 309 310 311 313]
            ratings =  [4 2 4 4 3 4 5 5 4 5 4 5 5 4 3 4 5 4 3 5 4 4 3 3 3 4 5 4 5 4 3 3 3 3 3 4 4
             1 4 3 4 3 4 5 4 3 4 1 4 5 3 1]
            m =  3.7115384615384617
            
            ...
            
            '''
            if np.isnan(m):
                m = 0  # to avoid empty array and nan value
            self.mu[n] = m  # mu[n] là điểm đánh giá trung bình của người dùng n
            # normalize
            '''
            chuẩn hóa cột điểm = cách lấy điểm cũ trừ đi điểm trung bình, lấy phần nguyên
            VD: Người dùng n cho 4 sản phẩm các điểm lần lượt là 3 4 5 6
            mean = 4.5
            Do đó, điểm lúc sau = nguyên(-1.5 -0.5 0.5 1.5) = -1 0 0 1
            '''
            self.Y_data_n[ids, 2] = ratings - self.mu[n]
            '''
            VD: n=0:
            self.Y_data_n[ids, 2] =  [ 1  0  0  0  0  1  0 -2  1  0 -1  1  1  1  1  1  0  1  0 -2  0  0  0  0
                  0 -1  0 -2  0  0  1  0 -1 -2 -1 -1  0  0  0 -1  1  0  1  1  0  1  0  1
                  0  0  0  0  1  0  1  0  1  1  0  0 -1  0  0  0  0  0  0  0  0  0 -2  0
                  0  0 -2  0  0  1  1  0  0  0  1  1  0  1  1  1 -1  0  1  0  0  0  1 -1
                 -1 -2 -2 -1  0  0  1  1 -2  1 -2  1  1  0  0  0  1 -2  0  0  0  1  0 -1
                  1  0  1  0 -2  0  0  0  0  0  1 -2  0 -2  0 -1 -2  0 -1  0  0 -1 -1  1
                  0  1  0  1 -1  0  0  0  0  0  0  0  0  0  1  1 -1  1  1  1  1  1  1  1
                  1  1  1  1  0  0  1  0  1  0  0  0  0  0  0  1  1  0  0  0  1  1  1  1
                  0  0  0  1  0  1  0  0  1  1  0  0  0  0 -1  0  0  1  0  0 -2  0  1  1
                  1 -1  0  0  0 -2  0 -1  0  1  0 -1  0  0  0  0  1 -2 -1 -1  1 -2  0  0
                  0  0 -1 -2 -1  0  0  1 -2 -2 -2  0 -2 -1  0 -2  0  1  1  1 -1  0]
            n=1:
            self.Y_data_n[ids, 2] =  [ 0 -1  0  0  0  0  1  1  0  1  0  1  1  0  0  0  1  0  0  1  0  0  0  0
                  0  0  1  0  1  0  0  0  0  0  0  0  0 -2  0  0  0  0  0  1  0  0  0 -2
                  0  1  0 -2]
            '''

    def loss(self):  # L(X,W)
        L = 0
        for i in range(self.n_ratings):  # duyet tat ca n_ratings =  90570 danh gia, n_ratings = s trong lý thuyết
            # user, item, rating
            n, m, rate = int(self.Y_data_n[i, 0]), int(self.Y_data_n[i, 1]), self.Y_data_n[i, 2]
            # n = người dùng thứ i
            # m = item thứ i
            # rate = đánh giá thứ i
            L += 0.5 * (rate - self.X[m, :].dot(self.W[:, n])) ** 2

        # take average
        L /= self.n_ratings
        # regularization, don't ever forget this
        L += 0.5 * self.lam * (np.linalg.norm(self.X, 'fro') + np.linalg.norm(self.W, 'fro'))
        return L

    def get_items_rated_by_user(self, user_id):  # lấy tất cả đánh giá của người dùng user_id
        """
        get all items which are rated by user user_id, and the corresponding ratings
        """
        ids = np.where(self.Y_data_n[:, 0] == user_id)[0]
        item_ids = self.Y_data_n[ids, 1].astype(np.int32)  # indices need to be integers
        ratings = self.Y_data_n[ids, 2]
        return (item_ids, ratings)  # trả về 1 tupple gồm 2 mảng: [ccác item đã đánh giá] và [số sao đánh giá]

    def get_users_who_rate_item(self, item_id):
        """
        get all users who rated item item_id and get the corresponding ratings
        """
        ids = np.where(self.Y_data_n[:, 1] == item_id)[0]
        user_ids = self.Y_data_n[ids, 0].astype(np.int32)
        ratings = self.Y_data_n[ids, 2]
        return (user_ids, ratings)

    def updateX(self):  #mỗi lần gọi thì cập nhật 1 hàng thôi
        for m in range(self.n_items):  # xét hàng thứ m (m chạy từ 1 đến M) của ma trận X (item thứ m)
            # Tại tất cả các dòng của item thứ m, lấy các người dùng và số sao mà người dùng đánh giá
            user_ids, ratings = self.get_users_who_rate_item(m)
            Wm = self.W[:, user_ids]  # lấy các cột của W ứng với các user_ids mà đã đánh giá item thứ m
            # gradient
            grad_xm = -(ratings - self.X[m, :].dot(Wm)).dot(Wm.T) / self.n_ratings + \
                      self.lam * self.X[m, :]
            self.X[m, :] -= self.learning_rate * grad_xm.reshape((self.K,))

    def updateW(self):  # xét cột thứ n (n chạy từ 1 đến N) của ma trận W (người dùng thứ n)
        for n in range(self.n_users):
            item_ids, ratings = self.get_items_rated_by_user(
                n)  # lấy các item và số sao đc đánh giá bởi người dùng thứ n
            Xn = self.X[item_ids, :]  # lấy các hàng của X ứng với các item_ids đã đc đánh giá bởi người dùng thứ n
            # gradient
            grad_wn = -Xn.T.dot(ratings - Xn.dot(self.W[:, n])) / self.n_ratings + \
                      self.lam * self.W[:, n]
            self.W[:, n] -= self.learning_rate * grad_wn.reshape((self.K,))

    def fit(self):
        self.normalize_Y()
        for it in range(self.max_iter):  # số lần dùng thuật toán descend gradient
            self.updateX()
            self.updateW()
            if (it + 1) % self.print_every == 0:    #cứ cập nhật 10 lần thì tính RMSE
                rmse_train = self.evaluate_RMSE(rate_test=self.Y_raw_data)    #đánh giá tập train
                print('iter =', it + 1, ', loss =', self.loss(), ', RMSE train =', rmse_train)

    def pred(self, u, i):  # dự đoán số sao mà người dùng u sẽ đánh giá sản phẩm i
        """
        predict the rating of user u for item i
        if you need the un
        """
        u = int(u)
        i = int(i)
        '''
            mu: ban đầu là 1 vector có n_objects số 0 = n_items hoặc n_users số 0 = 1682 hoặc 943 số 0
            trong quá trình chuẩn hóa sẽ được cập nhật dần dần
        '''
        if self.user_based:  # nếu chọn chế độ lọc cộng tác dựa trên người dùng
            bias = self.mu[u]
        else:
            bias = self.mu[i]  # bias là số sao trung bình mà sản phẩm i được đánh giá
        pred = self.X[i, :].dot(self.W[:, u]) + bias  # pred = X
        # truncate if results are out of range [0, 5]
        if pred < 0:
            return 0
        if pred > 5:
            return 5
        return pred

    def pred_for_user(self, user_id):  # điền nốt những ô mà user_id chưa đánh giá
        """
        predict ratings one user give all unrated items
        """
        ids = np.where(self.Y_data_n[:, 0] == user_id)[0]  # các dòng trong rate_train mà user_id đánh giá
        items_rated_by_u = self.Y_data_n[ids, 1].tolist()  # các item mà user_id đã đánh giá

        y_pred = self.X.dot(self.W[:, user_id]) + self.mu[
            user_id]  # dự đoán tất cả các ô mà user_id (bất kể đã đánh giá hay chưa)

        predicted_ratings = []
        for i in range(self.n_items):
            if i not in items_rated_by_u:
                predicted_ratings.append((i, y_pred[i]))  # chỉ thêm return những item mà user_id chưa đánh giá thôi

        return predicted_ratings

    def evaluate_RMSE(self, rate_test):
        n_tests = rate_test.shape[0]#n_test =  90570 hoặc 9430
        SE = 0  # squared error
        for n in range(n_tests):    #duyệt từng dòng một
            pred = self.pred(rate_test[n, 0], rate_test[n, 1])  #dự đoán số sao của người dùng thứ n
            SE += (pred - rate_test[n, 2]) ** 2  # RMSE = căn((tổng bình phương lỗi) / số test)

        RMSE = np.sqrt(SE / n_tests)
        return RMSE

    def write_X_W_to_csv(self):
        np.savetxt("X_MF.csv", self.X, delimiter=",")
        np.savetxt("W_MF.csv", self.W, delimiter=",")
        np.savetxt("B_MF.csv", +self.mu, delimiter=",")
        np.savetxt("Y_MF.csv", self.X.dot(self.W) + self.mu, delimiter=",")
        np.savetxt("Y_data_n_MF.csv", self.Y_data_n, delimiter=",")


def pred_for_user(nn):
    X_new = pd.read_csv('X_MF.csv', sep=',', header=None).values
    W_new = pd.read_csv('W_MF.csv', sep=',', header=None).values
    Y_new = pd.read_csv('Y_MF.csv', sep=',', header=None).values
    Y_data_n_new = pd.read_csv('Y_data_n_MF.csv', sep=',', header=None).values
    B_new = pd.read_csv('B_MF.csv', sep=',', header=None).values
    ids = np.where(Y_data_n_new[:, 0] == nn)[0]  # các dòng trong rate_train mà user_id đánh giá
    items_rated_by_u = Y_data_n_new[ids, 1].tolist()  # các item mà user_id đã đánh giá

    y_pred = X_new.dot(W_new[:, nn]) + B_new[nn]  # dự đoán tất cả các ô mà user_id (bất kể đã đánh giá hay chưa)

    predicted_ratings = []
    for i in range(int(np.max(Y_data_n_new[:, 1])) + 1):
        if i not in items_rated_by_u:
            predicted_ratings.append((i, y_pred[i]))  # chỉ thêm return những item mà user_id chưa đánh giá thôi

    return predicted_ratings


if __name__ == '__main__':
    # MovieLens 100k
    r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']

    ratings_base = pd.read_csv('ml-100k/ub.base', sep='\t', names=r_cols, encoding='latin-1')
    ratings_test = pd.read_csv('ml-100k/ub.test', sep='\t', names=r_cols, encoding='latin-1')

    rate_train = ratings_base.values
    rate_test = ratings_test.values

    '''
    rate_train và rate_test là 2 ma trận chứa các số được lưu ở trong 2 file ub.base và ub.test
    rate_train = 1 ma trận cỡ 90570 x 4
    rate_test = 1 ma trận cỡ 9430 x 4
    
    rate_train =  [[        1         1         5 874965758]
     [        1         2         3 876893171]
     [        1         3         4 878542960]
     ...
     [      943      1188         3 888640250]
     [      943      1228         3 888640275]
     [      943      1330         3 888692465]] 90570
     
    VD: Training có 90570 lượt đánh giá
    Người dùng 943 đánh giá phim 1330 3 sao
    Người dùng 1 đánh giá phim 2 3 sao
    ...
    
    rate_test =  [[        1        17         3 875073198]
     [        1        47         4 875072125]
     [        1        64         5 875072404]
     ...
     [      943       595         2 875502597]
     [      943       685         4 875502042]
     [      943      1011         2 875502560]] 9430
    Thực tế: Có 9434 lượt đánh giá
    Người dùng 943 đánh giá phim 595 2 sao
    Người dùng 1 đánh giá phim 47 4 sao
    ...
    '''

    # indices start from 0
    rate_train[:, :2] -= 1
    rate_test[:, :2] -= 1
    '''
    để chỉ số của ma trận bắt đầu từ 0, tại 2 cột đầu tiên của mỗi ma trận rate_train và rate_test
    trừ mỗi phần tử đi 1
    
    rate_train =  [[        0         0         5 874965758]
     [        0         1         3 876893171]
     [        0         2         4 878542960]
     ...
     [      942      1187         3 888640250]
     [      942      1227         3 888640275]
     [      942      1329         3 888692465]] 90570
    rate_test =  [[        0        16         3 875073198]
     [        0        46         4 875072125]
     [        0        63         5 875072404]
     ...
     [      942       594         2 875502597]
     [      942       684         4 875502042]
     [      942      1010         2 875502560]] 9430
    
    
    '''

    # # 1. chuẩn hoá dựa trên user:
    rs = MF(Y_data=rate_train, K=10, lam=.1, print_every=10,
            learning_rate=0.75, max_iter=100, user_based=1)
    '''
    K << M,N
    lam = hệ số lamda trong công thức hàm mất mát
    print_every = in ra loss fuction sau mỗi 10 iter
    learning_rate = hệ số eta trong công thức hàm mất mát
    max_iter = số lần cập nhật loss function tối đa
    user_based = 0 hoặc 1: Lọc cộng tác dựa trên nội dung film hay dựa trên người dùng
    s = số đánh giá của người dùng = số dòng trong ma trận rate_train
    '''

    rs.fit()
    # # evaluate on test data
    # RMSE = rs.evaluate_RMSE(rate_test=rate_test)
    # print('\nUser-based MF, RMSE =', RMSE)

    # # 2. chuẩn hoá dựa trên item:
    # rs = MF(rate_train, K=10, lam=0.1, print_every=10, learning_rate=0.75, max_iter=100, user_based=0)
    # rs.fit()
    # # evaluate on test data
    # RMSE = rs.evaluate_RMSE(rate_test)
    # print('\nItem-based MF, RMSE =', RMSE)

    # 3. không sử dụng regularization, tức lam = 0:
    # rs = MF(rate_train, K=2, lam=0, print_every=10, learning_rate=1, max_iter=100, user_based=0)
    # rs.fit()
    # # evaluate on test data
    # RMSE = rs.evaluate_RMSE(rate_test)
    # print('\nItem-based MF, RMSE =', RMSE)

    nn = 2
    print("Đánh giá trực tiếp: ", rs.pred_for_user(nn))

    # 4. Cập nhật tham số vào trong ma trận
    rs.write_X_W_to_csv()
    '''
    Đánh giá bằng pre_training:
    '''
    print("Đánh giá bằng pre_training: ", pred_for_user(nn))
