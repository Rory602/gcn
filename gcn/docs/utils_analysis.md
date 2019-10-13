# utils
对应文档中的数学公式为
```python
def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    
def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)
```


`normalize_adj`表示: <a href="https://www.codecogs.com/eqnedit.php?latex=D^{-1/2}AD^{-1/2}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?D^{-1/2}AD^{-1/2}" title="D^{-1/2}AD^{-1/2}" /></a>  
`preprocess_adj`表示: <a href="https://www.codecogs.com/eqnedit.php?latex=I_{N}&plus;D^{-\frac{1}{2}}&space;A&space;D^{-\frac{1}{2}}&space;\rightarrow&space;\tilde{D}^{-\frac{1}{2}}&space;\tilde{A}&space;\tilde{D}^{-\frac{1}{2}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?I_{N}&plus;D^{-\frac{1}{2}}&space;A&space;D^{-\frac{1}{2}}&space;\rightarrow&space;\tilde{D}^{-\frac{1}{2}}&space;\tilde{A}&space;\tilde{D}^{-\frac{1}{2}}" title="I_{N}+D^{-\frac{1}{2}} A D^{-\frac{1}{2}} \rightarrow \tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}}" /></a>  
其中:  
 &emsp;&emsp;<a href="https://www.codecogs.com/eqnedit.php?latex=\tilde{A}=A&plus;I_{N}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\tilde{A}=A&plus;I_{N}" title="\tilde{A}=A+I_{N}" /></a>  
 &emsp;&emsp;<a href="https://www.codecogs.com/eqnedit.php?latex=\tilde{D}_{i&space;i}=\sum_{j}&space;\tilde{A}_{i&space;j}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\tilde{D}_{i&space;i}=\sum_{j}&space;\tilde{A}_{i&space;j}" title="\tilde{D}_{i i}=\sum_{j} \tilde{A}_{i j}" /></a>
