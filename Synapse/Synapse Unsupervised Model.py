
"""
@author: Jodhani Zeel,Parth Dhameliya
"""

import numpy as np
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


st.title("Synapse Unsupervised Models")

uploaded_file = st.file_uploader("Choose a csv file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write(data)
    
if uploaded_file is not None:
    drop_column = st.sidebar.multiselect('X : Features (Selected will be dropped)', data.columns.to_list())
    
    X = data.drop(drop_column,axis = 1)
    st.header('X : Features')
    st.write(X)
   
if uploaded_file is not None:
    if st.sidebar.checkbox("Feature Normalization"):
        X = (X - np.mean(X))/np.std(X)
        st.header("X : Features (Normalized)")
        st.write(X)

class Kmeans:
    
    def initialize_var(self,X,K=3):
        X = np.array(X)
        m,n = X.shape
        c = np.random.randn(K,n)
        return X,c,K

    def assignment_move(self,X,c,K):
        m = X.shape[0]
        idx = np.zeros(m)
        for o in range(10):
            for i in range(m):
                temp = np.zeros(K)
                for j in range(K):
                    temp[j] = np.sum((X[i,:] - c[j,:]) ** 2) 
                    idx[i] = np.argmin(temp)
            for p in range(K):
                points = [X[j] for j in range(len(X)) if idx[j] == p]
                c[p] = np.mean(points, axis=0)
        return idx,c
    
    def test(self,X,K=3):
        self.X,c,self.K = self.initialize_var(X,K)
        self.idx,self.c = self.assignment_move(self.X,c,self.K)
        X_ = pd.DataFrame(self.X)
        idx_ = pd.DataFrame(self.idx)
        data = pd.concat([X_,idx_],axis =1)
        return self.c,data
    
    def plot_clusters(self,d):
        a={}
        if self.X.shape[1]==2:
            for i in range(2):
                a['a'+str(i+1)] = self.X[:,i:i+1]
            a['a1'] = np.reshape(a['a1'],(a['a1']).shape[0],)
            a['a2'] = np.reshape(a['a2'],(a['a2']).shape[0],)
            fig = go.Figure(data=go.Scatter(x=a['a1'], 
                              y=a['a2'], 
                              mode='markers',     
                              marker=dict(color=self.idx)
                              ))
            st.plotly_chart(fig)
        
        elif self.X.shape[1]==3:
            d.columns = ['x','y','z','l']
            fig = px.scatter_3d(d, x='x', y='y', z='z',color = 'l')
            st.plotly_chart(fig)
            
        elif self.X.shape[1]==3:
            print("Incomplete")
        else:
            st.error("Your data is in Higher Dimension state")

class PCA:
    
    def initialization(self,X):
        X = np.array(X)
        return X 
    
    def train(self,X):
        
        self.X = self.initialization(X)
        self.covariance_matrix = np.cov(X.T)
        self.u,s,v = np.linalg.svd(self.covariance_matrix)
        sum_s = np.sum(s)
        self.variance_exp= []
        k = 0
        for i in s:
            k = i+k
            variance = k/sum_s
            self.variance_exp.append(variance)
            
    def K_components(self,n=2):
            self.X= np.dot(self.X,self.u[:,:n])
            return self.X
    
    def variance_explained(self):
        return self.variance_exp

if uploaded_file is not None:               
        Algorithms = st.sidebar.selectbox(
            'Algorithm',
            ('None','K-means Clustering','Principal Component Analysis')
            )

        
if uploaded_file is not None:
    if Algorithms == 'K-means Clustering':
        k_value = st.sidebar.number_input('Enter K value',value = 3)
        
        train_button = st.sidebar.checkbox("Click Here for training")

        if train_button:
            d = Kmeans()
            c,data = d.test(X,k_value)
            st.subheader("Centroids")
            st.write(c)
            st.subheader("Clustering Data with labels")
            st.write(data)
            d.plot_clusters(data)
            
        #except : raise ValueError('graph not computed with NaN values or no. of K value exceeds try again')
    if Algorithms == 'Principal Component Analysis':
        k_value = st.sidebar.number_input('Enter K components value',value = 3)
        train_button = st.sidebar.checkbox("Click Here for training")
        
        if train_button:
            d = PCA()
            d.train(X)
            st.header('Variance Explained')
            st.markdown(d.variance_explained())
            st.info('Always Use Feature Normalization when applying PCA')
            X_pca = d.K_components(k_value)
            st.header('X : Feature (PCA)')
            st.write(X_pca)
            
   

    