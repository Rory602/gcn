# utils
## gcn
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

**解析:**  
`normalize_adj`表示: <a href="https://www.codecogs.com/eqnedit.php?latex=D^{-1/2}AD^{-1/2}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?D^{-1/2}AD^{-1/2}" title="D^{-1/2}AD^{-1/2}" /></a>  
`preprocess_adj`表示: <a href="https://www.codecogs.com/eqnedit.php?latex=I_{N}&plus;D^{-\frac{1}{2}}&space;A&space;D^{-\frac{1}{2}}&space;\rightarrow&space;\tilde{D}^{-\frac{1}{2}}&space;\tilde{A}&space;\tilde{D}^{-\frac{1}{2}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?I_{N}&plus;D^{-\frac{1}{2}}&space;A&space;D^{-\frac{1}{2}}&space;\rightarrow&space;\tilde{D}^{-\frac{1}{2}}&space;\tilde{A}&space;\tilde{D}^{-\frac{1}{2}}" title="I_{N}+D^{-\frac{1}{2}} A D^{-\frac{1}{2}} \rightarrow \tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}}" /></a>  
其中:  
 &emsp;&emsp;<a href="https://www.codecogs.com/eqnedit.php?latex=\tilde{A}=A&plus;I_{N}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\tilde{A}=A&plus;I_{N}" title="\tilde{A}=A+I_{N}" /></a>  
 &emsp;&emsp;<a href="https://www.codecogs.com/eqnedit.php?latex=\tilde{D}_{i&space;i}=\sum_{j}&space;\tilde{A}_{i&space;j}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\tilde{D}_{i&space;i}=\sum_{j}&space;\tilde{A}_{i&space;j}" title="\tilde{D}_{i i}=\sum_{j} \tilde{A}_{i j}" /></a>
 

参考文献  
Thomas N. Kipf, Max Welling, [Semi-Supervised Classification with Graph Convolutional Networks](http://arxiv.org/abs/1609.02907) (ICLR 2017)

```python

def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)
```
**解析**  
&emsp;`laplacian = sp.eye(adj.shape[0]) - adj_normalized`表示:  <a href="https://www.codecogs.com/eqnedit.php?latex=L=I_{n}-D^{-1&space;/&space;2}&space;W&space;D^{-1&space;/&space;2}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?L=I_{n}-D^{-1&space;/&space;2}&space;W&space;D^{-1&space;/&space;2}" title="L=I_{n}-D^{-1 / 2} W D^{-1 / 2}" /></a>  
&emsp;&emsp;其中:<a href="https://www.codecogs.com/eqnedit.php?latex=W" target="_blank"><img src="https://latex.codecogs.com/gif.latex?W" title="W" /></a>相当于gcn中的临接矩阵<a href="https://www.codecogs.com/eqnedit.php?latex=A" target="_blank"><img src="https://latex.codecogs.com/gif.latex?A" title="A" /></a>
`scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])`表示: <a href="https://www.codecogs.com/eqnedit.php?latex=\tilde{L}=2&space;L&space;/&space;\lambda_{\max&space;}-I_{n}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\tilde{L}=2&space;L&space;/&space;\lambda_{\max&space;}-I_{n}" title="\tilde{L}=2 L / \lambda_{\max }-I_{n}" /></a>  
&emsp;`2 * s_lap.dot(t_k_minus_one) - t_k_minus_two`表示: <a href="https://www.codecogs.com/eqnedit.php?latex=\bar{x}_{k}=2&space;\tilde{L}&space;\bar{x}_{k-1}-\bar{x}_{k-2}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\bar{x}_{k}=2&space;\tilde{L}&space;\bar{x}_{k-1}-\bar{x}_{k-2}" title="\bar{x}_{k}=2 \tilde{L} \bar{x}_{k-1}-\bar{x}_{k-2}" /></a>
