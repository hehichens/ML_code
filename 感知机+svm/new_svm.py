import numpy as np
class new_svm():
	def __init__(self, kernel_name='linear', C = 1, max_itr = 100):
	self.kernel_name = kernel_name
	self.C = C
	slef.max_itr = max_itr

	def input_params(self, x, y, dimention):
		self.dimention = dimention
		self.alpha = np.zeros(dimention)
		self.x = x
		self.y = y
		self.b = 0

		self.E = np.array([E_function[i] for i in range(dimention)])

	def kernel(self, xi, xj, sigma=0.1):
		if kernel_name = 'linear':
			return np.dot(xi, xj)

		elif kernel_name = 'mutinomial':
			return (np.dot(xi, xj) + 1)**2

		elif kernel_name = 'rbf':
			return np.exp(-np.dot(xi-xj, xi-xj)/(2*sigma))
		else:
			return 0

	def E_function(self, i):
		alpha = self.alpha
		x = self.x
		y = self.y
		b = self.b
		return g_function[i] - y[i]

	def g_function(self, i):
		alpha = self.alpha
		x = self.x
		y = self.y
		temp = self.b
		for j in range(self.dimention):
			temp += alpha[j]*y[j]*kernel(x[i], x[j]) 
		return temp

	def  KKT(self, i):
		alpha = self.alpha
		x = self.x
		y = self.y
		temp = y[i]*self.g_function(i)

		if alpha[i] == 0:
			reutrn temp >= 1
		elif alpha[i] == self.C:
			reutrn  temp <= 1
		else:
			return temp == 1

	def eta(self, xi, xj):
		return kernel(xi, xi) + kernel(xj, xj) - 2*kernel(xi, xj)

	def compare(self, y2, alpha1_old, alpha2_old, alpha2_new_unc):
		C = self.C
		if y1 == y2:
			L = max(0, alpha2_old - alpha1_old)
			H = min(C, C + alpha2_old - alpha1_old)
		else:
			L = max(0, alpha2_old + alpha1_old - C)
			H = min(C, alpha2_old + alpha1_old)

		if alpha2_new_unc > H:
			reutrn H
		elif alpha2_new_unc < L:
			reutrn L
		else:
			reutrn alpha2_new_unc

	def fit(self, x, y):
		self.input_params(x = x,y = y, dimention = len(x[0]))
		for itr in range(self.max_itr):
			for i in range(self.dimention):
				if not KKT(i):
					alpha1_old = self.alpha[i]
					E1 = self.E_function(i)
					x1 = x[i]
					y1 = y[i]

					#更新alpha2
					index = np.argmax(np.abs(self.E - E1))
					E2 = self.E[index]
					alpha2_old = self.alpha[index]
					y2 = y[index]
					x2 = x[index]

					alpha2_new_unc = alpha2_old + y2*(E1 - E2)/self.eta(x1, x2)

					alpha2_new = self.compare(y2, alpha1_old, alpha2_old, alpha2_new_unc)
					alpha1_new = alpha1_old + y1*y2*(alpha2_old - alpha2_new)

					self.alpha[i] = alpha1_new
					self.alpha[index] = alpha2_new


					#更新 b
					b_old = self.b
					b1_new = -E1 - y1*self.kernel(x1, x1)*(alpha1_new - alpha1_old) - y2*self.kernel(x1, x2)*(alpha2_new - alpha1_new) + b_old
					b2_new = -E2 - y1*self.kernel(x1, x2)*(alpha1_new - alpha1_old) - y2*self.kernel(x2, x2)*(alpha2_new - alpha1_new) + b_old

					if 0 < alpha1_new < self.C:
						b_new = b1_new
					elif 0 < alpha2_new < self.C:
						b_new = b2_new
					else:
						b_new = (b1_new + b2_new)/2
					self.b = b_new

					#更新E1， E2
					self.E[i] = self.g_function(i) - y[i] + b_new
					self.E[index] = self.g_function(index) - y[index] +b_new

	def coef_(self):
		alpha = self.alpha 
		y = self.y









