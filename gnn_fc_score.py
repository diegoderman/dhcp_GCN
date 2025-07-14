#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Main function to run the GNN model with age at scan as output and
functional connectivity as input.
This script reads in the preprocessed data, constructs the graph,
coarsens the graph, and trains a Graph Convolutional Neural Network (GCN)
to predict the age of neonates based on their functional connectivity features.
It uses a 5-fold cross-validation approach to evaluate the model's performance.
The results are saved to a CSV file and a scatter plot of predictions vs true values is generated


"""

import pandas as pd
from scipy.sparse import csr_matrix, load_npz
import numpy as np
import matplotlib.pyplot as plt
from lib import models, graph, coarsening
import config
#from lib.utils import train_val_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split
#from lib import dhcpy 

age_df = pd.read_csv('./resources/subjects.csv')
age_df = age_df.loc[:,['participant_id','scan_age']]

df_values_list = [pd.read_csv('/data/dderman/baby_ICA/gcn/data/sub-' + str(participant_id) + '.txt', header=None).values for participant_id in age_df['participant_id']]

df_values = np.stack(df_values_list,axis=0) 
#df_values = df_values[:,:,[0,1,2]].astype('float32')
print(df_values.shape) # nsubjs x nvertices x nfeatures
df_values = df_values.astype('float32')
n_features = df_values.shape[2]

graph_edges = pd.read_table('resources/edges1k.txt',names =['node1','node2'],encoding='UTF-16')
#num_nodes = graph_edges.iloc[:,0].max()+1
num_nodes=graph_edges.stack().max()

links = graph_edges.values
gg = np.random.permutation(links[:,1])
graph_edges['node3']=pd.DataFrame(gg)
adj_matrix = csr_matrix((np.ones(graph_edges.shape[0]),
            (graph_edges['node1'].values-1,
            graph_edges['node2'].values-1)),
           shape=(num_nodes,num_nodes))

r,c = adj_matrix.nonzero()
adj_matrix[c,r] = adj_matrix[r,c]

##################
# add my adjacency matrix from npz file

adj_matrix = load_npz('resources/adjacency_matrix_combined.npz').astype('float32')
r,c = adj_matrix.nonzero()
adj_matrix[c,r] = 1.0

cfg = coarsening.coarsen(adj_matrix, levels=9, self_connections=False)
graphs = cfg[0] # coarsened graphs
perm = cfg[1] # permutation vector for the coarsened graph


df_values = df_values[:, :, 0:num_nodes] # what is this for? redundant


data = coarsening.perm_data(df_values[:,:,:], perm) # permuts original data according to the coarsened graph
data = data.astype('float32')

L = [graph.laplacian(A, normalized=True).astype('float32') for A in graphs] # compute the Laplacian matrices for each coarsened graph

y = age_df['scan_age'].values # output variable: age at scan in weeks
#kf = KFold(n_splits=2,shuffle=True,random_state=98)

test_pred_all, test_labels_all, fold,  IDall = [],[],[],[]

params = vars(config.args)

X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.1, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.11, random_state=42) # 0.11 x 0.9 = 0.1

i = 1

#train_data = data[train_idx,:,:]
train_data = X_train
train_data = train_data.reshape([train_data.shape[0],-1])
scaler = StandardScaler()
train_data_trans = scaler.fit_transform(train_data)
train_data_trans = train_data_trans.reshape([train_data.shape[0],-1,n_features])
#train_labels = y[train_idx].astype('float32')
train_labels = y_train.astype('float32')

#valid_data = data[valid_ind,:,:]
valid_data = X_val
valid_data = valid_data.reshape([valid_data.shape[0],-1])
valid_data_trans = scaler.transform(valid_data)
valid_data_trans = valid_data_trans.reshape([valid_data.shape[0],-1,n_features])
#valid_labels = y[valid_ind].astype('float32')
valid_labels = y_val.astype('float32')

#test_data = data[test_idx,:,:]
test_data = X_test
test_data = test_data.reshape([test_data.shape[0],-1])
test_data_trans = scaler.transform(test_data)
test_data_trans = test_data_trans.reshape([test_data.shape[0],-1,n_features])
#test_labels = y[test_idx].astype('float32')
test_labels = y_test.astype('float32')

#mean_predict = np.tile(train_labels.mean(), valid_labels.shape)
#print('baseline MSE: ', mean_squared_error(valid_labels,mean_predict))

model = models.cgcnn(L, **params)
model.fit(train_data=train_data_trans,train_labels=train_labels,val_data=valid_data_trans,val_labels=valid_labels)

y_pred = model.predict(data=test_data_trans)

plt.scatter(test_labels, y_pred)

test_pred_all.append(y_pred)
test_labels_all.append(test_labels)
IDall.append(age_df.ID[test_ind].tolist())
fold.append([i]*len(y_pred))



flatten = lambda z: [x for y in z for x in y]
pred_array = flatten(test_pred_all)
true_array = flatten(test_labels_all)
IDall = flatten(IDall)
fold = flatten(fold)

df_out = pd.DataFrame({'ID':IDall, 'true':true_array,'pred':pred_array,'fold':fold})

output_fn = 'predicted_age'
df_out.to_csv(output_fn + '.csv')

plt.xlim([20,45])
plt.ylim([20,45])
plt.xlabel('true age (weeks)')
plt.ylabel('predicted age (weeks)')
plt.savefig('test.png')
