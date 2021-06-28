import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import matplotlib.pyplot as plt
import math
import random
import csv
import re
# Global variables
phase = "train"  # phase can be set to either "train" or "eval"

""" 
You are allowed to change the names of function "arguments" as per your convenience, 
but it should be meaningful.

E.g. y, y_train, y_test, output_var, target, output_label, ... are acceptable
but abc, a, b, etc. are not.

"""

#The files used in the program are 'train.csv', 'dev.csv' and 'test.csv'
def one_hot_encoder(phi, phi_dev, phi_test):
	#one hot encoding name, owner, fuel, transmission and seller type features across train, dev and test set
	phi['train'] = 1
	phi_test['train'] = 0
	phi_dev['train'] = 2
	combined = pd.concat([phi, phi_dev, phi_test])
	df_name = pd.get_dummies(combined['name'], prefix='name', drop_first=True)
	df_owner = pd.get_dummies(combined['owner'], prefix='owner', drop_first=True)
	df_sell = pd.get_dummies(combined['seller_type'], prefix='sell', drop_first=True)
	df_trans = pd.get_dummies(combined['transmission'], prefix='trans', drop_first=True)
	df_fuel = pd.get_dummies(combined['fuel'], prefix='fuel', drop_first=True)
	combined = pd.concat([combined,df_name], axis = 1)
	combined = pd.concat([combined,df_owner], axis = 1)
	combined = pd.concat([combined,df_sell], axis = 1)
	combined = pd.concat([combined,df_trans], axis = 1)
	combined = pd.concat([combined,df_fuel], axis = 1)

	combined = combined.drop('name', axis=1)
	combined = combined.drop('owner', axis=1)
	combined = combined.drop('seller_type', axis=1)
	combined = combined.drop('transmission', axis=1)
	combined = combined.drop('fuel', axis=1)

	df_train = combined[combined["train"] ==1]
	df_test = combined[combined["train"] ==0]
	df_dev = combined[combined["train"] ==2]
	df_train.drop(["train"], axis = 1, inplace=True)
	df_test.drop(["train"], axis = 1, inplace=True)
	df_dev.drop(["train"], axis = 1, inplace=True)


	return df_train, df_dev, df_test
	

def get_features(file_path):
	# Given a file path , return feature matrix and target labels
	data = pd.read_csv(file_path, index_col=0, parse_dates=True)
	data = data.fillna(method = "bfill")

	#pre-processing mileage, engine, name and torque
	for i in range(len(data)):
		data.loc[i, "mileage"] = float(data.loc[i, "mileage"].split()[0])
		data.loc[i, "engine"] = float(data.loc[i, "engine"].split()[0])
		data.loc[i, "name"] = data.loc[i, "name"].split()[0]


	for i in range(len(data)):
		x = re.findall("k", data.loc[i,'torque'])
		y = re.findall("N", data.loc[i,'torque'])

		if x and y:
			data.loc[i, "torque"] = data.loc[i, "torque"].split("@")[0]
			data.loc[i, "torque"] = data.loc[i, "torque"].split("N")[0]
			data.loc[i, "torque"] = data.loc[i, "torque"].split()[0]
			data.loc[i, "torque"] = data.loc[i, "torque"].split("n")[0]
			data.loc[i, "torque"] = data.loc[i, "torque"].split("(")[0]

		elif x:
			data.loc[i, "torque"] = data.loc[i, "torque"].split("@")[0]
			data.loc[i, "torque"] = data.loc[i, "torque"].split()[0]
			data.loc[i, "torque"] = data.loc[i, "torque"].split("k")[0]
			data.loc[i, "torque"] = data.loc[i, "torque"].split("(")[0]
			data.loc[i, "torque"] = float(data.loc[i, "torque"]) * 9.8
		else:
			data.loc[i, "torque"] = data.loc[i, "torque"].split("@")[0]
			data.loc[i, "torque"] = data.loc[i, "torque"].split("N")[0]
			data.loc[i, "torque"] = data.loc[i, "torque"].split()[0]
			data.loc[i, "torque"] = data.loc[i, "torque"].split("n")[0]
			data.loc[i, "torque"] = data.loc[i, "torque"].split("(")[0]

	data["torque"] = data["torque"].astype(np.float64)

	#normalisation of values
	data.year = (data.year - data.year.min()) / \
		(data.year.max() - data.year.min())
	data.km_driven = (data.km_driven - data.km_driven.min()) / \
		(data.km_driven.max() - data.km_driven.min())
	data.engine = (data.engine - data.engine.min()) / \
		(data.engine.max() - data.engine.min())
	data.max_power = (data.max_power - data.max_power.min()) / \
		(data.max_power.max() - data.max_power.min())
	data.mileage = (data.mileage - data.mileage.min()) / \
		(data.mileage.max() - data.mileage.min())
	data.torque = (data.torque - data.torque.min()) / \
		(data.torque.max() - data.torque.min())
	data.seats = (data.seats - data.seats.min()) / \
		(data.seats.max() - data.seats.min())

	#get features
	phi = data.loc[:, data.columns != 'selling_price']

	#get labels
	if 'selling_price' in data.columns:
		y = data.selling_price
		y = y.multiply(0.000001)  # process selling price in hundred thousands
		return phi, y

	else:
		return phi

def get_features_basis(file_path):
	# Given a file path , return feature matrix and target labels
	data = pd.read_csv(file_path, index_col=0, parse_dates=True)
	data = data.fillna(method = "bfill")

	#pre-processing mileage, engine, name and torque
	for i in range(len(data)):
		data.loc[i, "mileage"] = float(data.loc[i, "mileage"].split()[0])
		data.loc[i, "engine"] = float(data.loc[i, "engine"].split()[0])
		data.loc[i, "name"] = data.loc[i, "name"].split()[0]


	for i in range(len(data)):
		x = re.findall("k", data.loc[i,'torque'])
		y = re.findall("N", data.loc[i,'torque'])

		if x and y:
			data.loc[i, "torque"] = data.loc[i, "torque"].split("@")[0]
			data.loc[i, "torque"] = data.loc[i, "torque"].split("N")[0]
			data.loc[i, "torque"] = data.loc[i, "torque"].split()[0]
			data.loc[i, "torque"] = data.loc[i, "torque"].split("n")[0]
			data.loc[i, "torque"] = data.loc[i, "torque"].split("(")[0]

		elif x:
			data.loc[i, "torque"] = data.loc[i, "torque"].split("@")[0]
			data.loc[i, "torque"] = data.loc[i, "torque"].split()[0]
			data.loc[i, "torque"] = data.loc[i, "torque"].split("k")[0]
			data.loc[i, "torque"] = data.loc[i, "torque"].split("(")[0]
			data.loc[i, "torque"] = float(data.loc[i, "torque"]) * 9.8
		else:
			data.loc[i, "torque"] = data.loc[i, "torque"].split("@")[0]
			data.loc[i, "torque"] = data.loc[i, "torque"].split("N")[0]
			data.loc[i, "torque"] = data.loc[i, "torque"].split()[0]
			data.loc[i, "torque"] = data.loc[i, "torque"].split("n")[0]
			data.loc[i, "torque"] = data.loc[i, "torque"].split("(")[0]

	data["torque"] = data["torque"].astype(np.float64)

	#normalisation of values
	data.year = (data.year - data.year.min()) / \
		(data.year.max() - data.year.min())
	data.km_driven = (data.km_driven - data.km_driven.min()) / \
		(data.km_driven.max() - data.km_driven.min())
	data.engine = (data.engine - data.engine.min()) / \
		(data.engine.max() - data.engine.min())
	data.max_power = (data.max_power - data.max_power.min()) / \
		(data.max_power.max() - data.max_power.min())
	data.mileage = (data.mileage - data.mileage.min()) / \
		(data.mileage.max() - data.mileage.min())
	data.torque = (data.torque - data.torque.min()) / \
		(data.torque.max() - data.torque.min())
	data.seats = (data.seats - data.seats.min()) / \
		(data.seats.max() - data.seats.min())

	#applying basis functions
	data["interaction"] = data.year * data.km_driven
	data["max_power_squared"] = data.max_power * data.max_power
	data["max_power_cube"] = data.max_power * data.max_power * data.max_power

	#get features
	phi = data.loc[:, data.columns != 'selling_price']

	#get labels
	if 'selling_price' in data.columns:
		y = data.selling_price
		y = y.multiply(0.000001)  # process selling price in hundred thousands
		return phi, y

	else:
		return phi


def compute_RMSE(phi, w, y):
	# Root Mean Squared Error
	phi_n = phi.to_numpy()
	phi_n = np.array(phi_n, dtype=np.float64)
	w_n = w.to_numpy()
	w_n = np.array(w_n, dtype=np.float64)
	y = y.multiply(1000000)
	y_n = y.to_numpy()
	y_n = np.array(y_n, dtype=np.float64)
	y_n = y_n.reshape(len(y), 1)

	sol = np.dot(phi_n, w_n)
	sol = sol*1000000
	error_matrix = sol - y_n

	#getting the mean squared error using formula
	error = (np.linalg.norm(error_matrix)/math.sqrt(len(y)))


	return error


def generate_output(phi_test, w):
	# writes a file (output.csv) containing target variables in required format for Submission.
	phi_n = phi_test.to_numpy()
	phi_n = np.array(phi_n, dtype=np.float64)
	w_n = w.to_numpy()
	w_n = np.array(w_n, dtype=np.float64)
	sol_n = np.dot(phi_n, w_n)
	sol = pd.DataFrame(sol_n)
	sol = sol*1000000

	sol.to_csv('output.csv')


def closed_soln(phi, y):
	# Function returns the solution w for Xw=y.
	phi_n = phi.to_numpy()
	phi_n = np.array(phi_n, dtype=np.float64)
	y_n = y.to_numpy()
	y_n = np.array(y_n, dtype=np.float64)
	csol = np.linalg.pinv(phi_n).dot(y_n)
	return pd.DataFrame(csol)  # return the dot product as a data frame


def gradient_descent(phi, y, phi_dev, y_dev):
	# Implement gradient_descent using Mean Squared Error Loss
	# You may choose to use the dev set to determine point of convergence

	#setting initial w vector as a dataframe and a numpy array
	w_n = np.zeros((len(phi.columns), 1))
	w = pd.DataFrame(w_n)

	#getting the first term in the gradient, -2X'Y
	first_term = phi.transpose()
	first_term = first_term.dot(y)
	first_term = first_term.multiply(-2)
	first_term_n = first_term.to_numpy()
	first_term_n = first_term_n.flatten()

	# u matrix is X'X to be used later
	u = phi.transpose()
	u = u.dot(phi)
	u = u.astype(np.float64)
	u_n = u.to_numpy()
	rmse_i = compute_RMSE(phi_dev, w, y_dev)
	t = 0

	while True:
		#calculating the second term in the gradient, 2X'XW as a numpy array
		sec_term = np.dot(u_n, w_n)
		sec_term = sec_term.astype(np.float64)
		sec_term = sec_term*2
		sec_term = sec_term.flatten()

		#getting the gradient as a dataframe
		gradient_n = first_term_n + sec_term
		gradient = pd.DataFrame(gradient_n)
		#changing the w
		w = w - gradient.multiply(0.000001)
		w_n = w.to_numpy()
		rmse_current = compute_RMSE(phi_dev, w, y_dev)

		#stopping criteria when the error increases on the dev set for a certain number of epochs
		if rmse_current < rmse_i and t < 10:
				rmse_i = rmse_current
				w_star = w
				t = 0
		elif rmse_current > rmse_i and t < 10:
			t = t+1

		else:
			break

	return w_star


def sgd(phi, y, phi_dev, y_dev):
	# Implement stochastic gradient_descent using Mean Squared Error Loss
	# You may choose to use the dev set to determine point of convergence
	w_n = np.zeros((len(phi.columns), 1))
	w = pd.DataFrame(w_n)
	rmse_i = compute_RMSE(phi_dev, w, y_dev)
	t = 0
	while True:
		#getting random point from training set
		ra = random.randint(0, len(phi)-1)
		phi_p = phi.iloc[ra, :]
		phi_p = pd.DataFrame(phi_p)
		y_p = y.iloc[ra]

		#applying gradient descent using the random point
		first_term = phi_p.multiply(y_p)
		first_term = first_term.multiply(-2)
		first_term_n = first_term.to_numpy()
		first_term_n = first_term_n.flatten()
		phi_p = phi_p.transpose()
		u = phi_p.transpose()
		u = u.dot(phi_p)
		u = u.astype(np.float64)
		u_n = u.to_numpy()
		sec_term = np.dot(u_n, w_n)
		sec_term = sec_term.astype(np.float64)
		sec_term = sec_term*2
		sec_term = sec_term.flatten()
		g_n = first_term_n + sec_term
		g = pd.DataFrame(g_n)
		g = g.astype(np.float64)
		w = w - g.multiply(0.001)
		rmse_current = compute_RMSE(phi_dev, w, y_dev)
		w_n = w.to_numpy()
		if rmse_current < rmse_i and t < 80:
				rmse_i = rmse_current
				w_star = w
				t = 0
		elif rmse_current > rmse_i and t < 80:
			t = t+1

		else:
			break

	return w_star


def pnorm(phi, y, phi_dev, y_dev, p):
	# Implement gradient_descent with p-norm regularisation using Mean Squared Error Loss
	# You may choose to use the dev set to determine point of convergence
	if p == 2:
		lam = 50
		w_n = np.zeros((len(phi.columns), 1))
		w = pd.DataFrame(w_n)
		first_term = phi.transpose()
		first_term = first_term.dot(y)
		first_term = first_term.multiply(-2)
		first_term_n = first_term.to_numpy()
		first_term_n = first_term_n.flatten()
		u = phi.transpose()
		u = u.dot(phi)
		u = u.astype(np.float64)
		u_n = u.to_numpy()
		while True:
			sec_term = np.dot(u_n, w_n)
			sec_term = sec_term.astype(np.float64)
			sec_term = sec_term*2
			sec_term = sec_term.flatten()
			#additional third term in gradient due to regularization
			thd_term = 2*lam*w_n
			thd_term = thd_term.astype(np.float64)
			thd_term = thd_term.flatten()
			g_n = first_term_n + sec_term + thd_term
			g = pd.DataFrame(g_n)
			w = w - g.multiply(0.000001)
			g2 = g.multiply(0.000001)
			g_n2 = g2.to_numpy()
			norm = np.linalg.norm(g_n2)
			w_n = w.to_numpy()
			#stopping criteria when stepsize becomes very small
			if norm < 0.000002:
				break
			else:
				continue

		return w

	elif p == 4:
		lam = 40
		w_n = np.zeros((len(phi.columns), 1))
		w = pd.DataFrame(w_n)
		first_term = phi.transpose()
		first_term = first_term.dot(y)
		first_term = first_term.multiply(-2)
		first_term_n = first_term.to_numpy()
		first_term_n = first_term_n.flatten()
		u = phi.transpose()
		u = u.dot(phi)
		u = u.astype(np.float64)
		u_n = u.to_numpy()
		while True:
			sec_term = np.dot(u_n, w_n)
			sec_term = sec_term.astype(np.float64)
			sec_term = sec_term*2
			sec_term = sec_term.flatten()
			#additional third term in gradient due to regularization
			thd_term = w
			thd_term = thd_term*thd_term*thd_term*4*lam
			thd_term_n = thd_term.to_numpy()
			thd_term_n = thd_term_n.astype(np.float64)
			thd_term_n = thd_term_n.flatten()
			g_n = first_term_n + sec_term + thd_term_n
			g = pd.DataFrame(g_n)
			w = w - g.multiply(0.000001)
			g2 = g.multiply(0.000001)
			g_n2 = g2.to_numpy()
			norm = np.linalg.norm(g_n2)
			w_n = w.to_numpy()
			if norm < 0.00002:
				break
			else:
				continue
			
		return w


def plot():
	phi, y = get_features('train.csv')
	phi_dev, y_dev = get_features('dev.csv')
	phi_test = get_features('test.csv')
	phi, phi_dev, phi_test = one_hot_encoder(phi, phi_dev, phi_test)
	x = [2000, 2500, 3000, len(phi)]
	y_p = []
	for i in range(3):
		w = pnorm(phi[:(2000 + (500*i))], y[:(2000 + (500*i))], phi_dev, y_dev, 2)
		y_p.append(compute_RMSE(phi_dev, w, y_dev))

	w = pnorm(phi, y, phi_dev, y_dev, 2)
	y_p.append(compute_RMSE(phi_dev, w, y_dev))
	plt.plot(x, y_p)
	plt.xlabel('Number of Instances from train.csv')
	plt.ylabel('RMSE on dev.csv')
	plt.title('Q4 - RMSE vs Number of instances')
	plt.show()

#Now follow the functions can be implemented to generate the output submitted on Kaggle
def one_hot_encoder_for_task6(phi, phi_dev, phi_test):
	#one hot encoding name, owner, fuel, transmission, engine, seats and seller type features across train, dev and test set
	phi['train'] = 1
	phi_test['train'] = 0
	phi_dev['train'] = 2
	combined = pd.concat([phi, phi_dev, phi_test])
	df_name = pd.get_dummies(combined['name'], prefix='name', drop_first=True)
	df_owner = pd.get_dummies(combined['owner'], prefix='owner', drop_first=True)
	df_sell = pd.get_dummies(combined['seller_type'], prefix='sell', drop_first=True)
	df_trans = pd.get_dummies(combined['transmission'], prefix='trans', drop_first=True)
	df_fuel = pd.get_dummies(combined['fuel'], prefix='fuel', drop_first=True)
	df_engine = pd.get_dummies(combined['engine'], prefix='engine', drop_first=True)
	df_seats = pd.get_dummies(combined['seats'], prefix='seats', drop_first=True)

	combined = pd.concat([combined,df_name], axis = 1)
	combined = pd.concat([combined,df_owner], axis = 1)
	combined = pd.concat([combined,df_sell], axis = 1)
	combined = pd.concat([combined,df_trans], axis = 1)
	combined = pd.concat([combined,df_fuel], axis = 1)
	combined = pd.concat([combined,df_engine], axis = 1)
	combined = pd.concat([combined,df_seats], axis = 1)

	combined = combined.drop('name', axis=1)
	combined = combined.drop('owner', axis=1)
	combined = combined.drop('seller_type', axis=1)
	combined = combined.drop('transmission', axis=1)
	combined = combined.drop('fuel', axis=1)
	combined = combined.drop('engine', axis=1)
	combined = combined.drop('seats', axis=1)

	df_train = combined[combined["train"] ==1]
	df_test = combined[combined["train"] ==0]
	df_dev = combined[combined["train"] ==2]
	df_train.drop(["train"], axis = 1, inplace=True)
	df_test.drop(["train"], axis = 1, inplace=True)
	df_dev.drop(["train"], axis = 1, inplace=True)


	return df_train, df_dev, df_test

def get_features_basis_for_task6(file_path):
	# Given a file path , return feature matrix and target labels
	data = pd.read_csv(file_path, index_col=0, parse_dates=True)
	data = data.fillna(method = "bfill")

	#pre-processing mileage, engine, name and torque
	for i in range(len(data)):
		data.loc[i, "mileage"] = float(data.loc[i, "mileage"].split()[0])
		#data.loc[i, "engine"] = float(data.loc[i, "engine"].split()[0])
		data.loc[i, "name"] = data.loc[i, "name"].split()[0]


	for i in range(len(data)):
		x = re.findall("k", data.loc[i,'torque'])
		y = re.findall("N", data.loc[i,'torque'])

		if x and y:
			data.loc[i, "torque"] = data.loc[i, "torque"].split("@")[0]
			data.loc[i, "torque"] = data.loc[i, "torque"].split("N")[0]
			data.loc[i, "torque"] = data.loc[i, "torque"].split()[0]
			data.loc[i, "torque"] = data.loc[i, "torque"].split("n")[0]
			data.loc[i, "torque"] = data.loc[i, "torque"].split("(")[0]

		elif x:
			data.loc[i, "torque"] = data.loc[i, "torque"].split("@")[0]
			data.loc[i, "torque"] = data.loc[i, "torque"].split()[0]
			data.loc[i, "torque"] = data.loc[i, "torque"].split("k")[0]
			data.loc[i, "torque"] = data.loc[i, "torque"].split("(")[0]
			data.loc[i, "torque"] = float(data.loc[i, "torque"]) * 9.8
		else:
			data.loc[i, "torque"] = data.loc[i, "torque"].split("@")[0]
			data.loc[i, "torque"] = data.loc[i, "torque"].split("N")[0]
			data.loc[i, "torque"] = data.loc[i, "torque"].split()[0]
			data.loc[i, "torque"] = data.loc[i, "torque"].split("n")[0]
			data.loc[i, "torque"] = data.loc[i, "torque"].split("(")[0]

	data["torque"] = data["torque"].astype(np.float64)


	#normalisation of values
	data.year = (data.year - data.year.min()) / \
		(data.year.max() - data.year.min())
	data.km_driven = (data.km_driven - data.km_driven.min()) / \
		(data.km_driven.max() - data.km_driven.min())
	data.max_power = (data.max_power - data.max_power.min()) / \
		(data.max_power.max() - data.max_power.min())
	data.mileage = (data.mileage - data.mileage.min()) / \
		(data.mileage.max() - data.mileage.min())
	data.torque = (data.torque - data.torque.min()) / \
		(data.torque.max() - data.torque.min())

	data["max_power2"] = data.max_power*data.max_power
	data["max_power3"] = data.max_power*data.max_power*data.max_power
	data["inter"] = data.km_driven * data.year
	#get features
	phi = data.loc[:, data.columns != 'selling_price']

	#get labels
	if 'selling_price' in data.columns:
		y = data.selling_price
		y = y.multiply(0.000001)  # process selling price in hundred thousands
		return phi, y

	else:
		return phi

def pnorm_for_task6(phi, y, phi_dev, y_dev, p):
	# Implement gradient_descent with p-norm regularisation using Mean Squared Error Loss
	# You may choose to use the dev set to determine point of convergence
	if p == 4:
		lam = 150
		w_n = np.zeros((len(phi.columns), 1))
		w = pd.DataFrame(w_n)
		first_term = phi.transpose()
		first_term = first_term.dot(y)
		first_term = first_term.multiply(-2)
		first_term_n = first_term.to_numpy()
		first_term_n = first_term_n.flatten()
		u = phi.transpose()
		u = u.dot(phi)
		u = u.astype(np.float64)
		u_n = u.to_numpy()
		for i in range(40000):
			sec_term = np.dot(u_n, w_n)
			sec_term = sec_term.astype(np.float64)
			sec_term = sec_term*2
			sec_term = sec_term.flatten()
			#additional third term in gradient due to regularization
			thd_term = w
			thd_term = thd_term*thd_term*thd_term*4*lam
			thd_term_n = thd_term.to_numpy()
			thd_term_n = thd_term_n.astype(np.float64)
			thd_term_n = thd_term_n.flatten()
			g_n = first_term_n + sec_term + thd_term_n
			g = pd.DataFrame(g_n)
			w = w - g.multiply(0.000001)
			g2 = g.multiply(0.000001)
			g_n2 = g2.to_numpy()
			norm = np.linalg.norm(g_n2)
			w_n = w.to_numpy()

		return w

def task6():
	phi, y = get_features_basis_for_task6('train.csv')
	phi_dev, y_dev = get_features_basis_for_task6('dev.csv')
	phi_test = get_features_basis_for_task6('test.csv')
	phi, phi_dev, phi_test = one_hot_encoder_for_task6(phi, phi_dev, phi_test)
	
	w = pnorm_for_task6(phi, y, phi_dev, y_dev, 4)
	
	generate_output(phi_test, w)



""" 
The following steps will be run in sequence by the autograder.
"""

######## Task 1 #########
def main():
	phase = "train"
	phi, y = get_features('train.csv')
	phase = "eval"
	phi_dev, y_dev = get_features('dev.csv')
	phi_test = get_features('test.csv')
	phi, phi_dev, phi_test = one_hot_encoder(phi, phi_dev, phi_test)
	w1 = closed_soln(phi, y)
	w2 = gradient_descent(phi, y, phi_dev, y_dev)
	r1 = compute_RMSE(phi_dev, w1, y_dev)
	r2 = compute_RMSE(phi_dev, w2, y_dev)
	print('1a: ')
	print(abs(r1-r2))
	w3 = sgd(phi, y, phi_dev, y_dev)
	r3 = compute_RMSE(phi_dev, w3, y_dev)
	print('1c: ')
	print(abs(r2-r3))

	######## Task 2 #########
	w_p2 = pnorm(phi, y, phi_dev, y_dev, 2)
	w_p4 = pnorm(phi, y, phi_dev, y_dev, 4)
	r_p2 = compute_RMSE(phi_dev, w_p2, y_dev)
	r_p4 = compute_RMSE(phi_dev, w_p4, y_dev)
	print('2: pnorm2')
	print(r_p2)
	print('2: pnorm4')
	print(r_p4)

	######## Task 3 #########
	phase = "train"
	phi_basis, y = get_features_basis('train.csv')
	phase = "eval"
	phi_dev, y_dev = get_features_basis('dev.csv')
	phi_test = get_features_basis('test.csv')
	phi_basis, phi_dev, phi_test = one_hot_encoder(phi_basis, phi_dev, phi_test)
	w_basis = pnorm(phi_basis, y, phi_dev, y_dev, 2)
	rmse_basis = compute_RMSE(phi_dev, w_basis, y_dev)
	print('Task 3: basis')
	print(rmse_basis)

	######## Task 6: The following code can be run to generate the output submitted on Kaggle ######
	#task6()


main()
#plot()



