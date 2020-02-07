class my_svm():
    def __init__(self, max_itr=100, kernel_name='linear'):
        self.kernel_name = kernel_name
        self.max_itr = max_itr
        
    def input_param(self, features, labels):
        self.b = 0
        self.m, self.n = features.shape
        self.X = features
        self.y = labels
        
        self.alp = np.zeros(self.m)
        self.C = 1
        self.E = np.array([self.E_function(i) for i in range(self.m)])
        
    #预测值
    def g_function(self, i):
        temp = 0
        xi = self.X[i]
        for j in range(self.m):
            xj = self.X[j]
            temp += self.alp[j]*self.y[j]*self.kernel(xi, xj)
        return temp + self.b
    
    #计算Ei值
    def E_function(self, i):
        return self.g_function(i) - self.y[i]
        
    #核函数
    def kernel(self, x, z):
        if self.kernel_name == 'linear':
            return np.dot(x, z)
        elif self.kernel_name == 'poly':
            return (np.dot(x, z) + 1)**2
        elif self.kernel_name == 'gaussian':
            #sigma = np.std(self.x)
            return np.exp(-(x-z)**2)
        return 0
    
    #mu函数
    def mu_function(self, x1, x2):
        return self.kernel(x1, x1) + self.kernel(x2, x2) - 2*self.kernel(x1, x2)
    
    #KKT条件
    def KKT(self, i):
        temp = self.y[i]*self.g_function(i)
        alp = self.alp[i]
        if alp == 0:
            return temp >= 1
        elif alp == self.C:
            return temp <= 1
        else:
            return temp == 1
    
    def fit(self, X, y):
        self.input_param(X, y)

        for itr in range(self.max_itr):
            for i in range(self.m):
                if not self.KKT(i):
                    b_old = self.b
                    alp1_old = self.alp[i]
                    y1 = self.y[i]
                    x1 = self.X[i]
                    E1 = self.E[i]

                    #alp2的选择
                    j = np.argmax(np.abs(self.E - E1))
                    alp2_old = self.alp[j]
                    y2 = self.y[j]
                    E2 = self.E[j]
                    x2 = self.X[j]
                    
                    mu = self.mu_function(x1, x2)
                    alp2_new_unc = alp2_old + y2*(E1 - E2)/mu
                    
                    #限制条件L， H
                    if y1 != y2:
                        L = max(0, alp2_old - alp1_old)
                        H = min(self.C, self.C + alp2_old + alp1_old)
                    else:
                        L = max(0, alp2_old + alp1_old - self.C)
                        H = min(self.C, alp2_old + alp1_old)
                    
                    #是否更新alp2
                    if alp2_new_unc > H:
                        alp2_new = H
                    elif alp2_new_unc < L:
                        alp2_new = L
                    else:
                        alp2_new = alp2_new_unc
                    
                    alp1_new = alp1_old +y1*y2*(alp2_old - alp2_new) # 更新alp1
                    
                    b1_new = -E1 - y1*self.kernel(x1, x1)*(alp1_new - alp1_old) - y2*self.kernel(x2, x1)*(alp2_new - alp2_old) + b_old
                    b2_new = -E2 - y1*self.kernel(x1, x2)*(alp1_new - alp1_old) - y2*self.kernel(x2, x2)*(alp2_new - alp2_old) + b_old
                    
                    if 0 < alp1_new < self.C:
                        b_new = b1_new
                    elif 0 < alp2_new < self.C:
                        b_new = b2_new
                    else:
                        b_new = (b1_new + b2_new)/2
                    
                    #更新参数
                    self.alp[i] = alp1_new
                    self.alp[j] = alp2_new
                    self.b = b_new

                    #重新计算E，更新！！不是用E1，E2， 因为alp，b更新了， 所以可以得到新的E
                    self.E[i] = self.E_function(i)
                    self.E[j] = self.E_function(j)
                    # print('i am here')

        return 'training done !'

    def predict(self, X):
        res = []
        for x in X:
            #temp从b开始累加
            temp = self.b
            for i in range(self.m):
                temp += self.alp[i]*self.y[i]*self.kernel(x, self.X[i])
            r = 1 if temp > 0 else -1
            res.append(r)
        return np.array(res)

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

    def weight(self):
        yx = self.y.reshape(-1, 1) * self.X
        self.w = np.dot(yx.T, self.alp)
        return self.w