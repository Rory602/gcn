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
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple 			representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)
```

**解析:**  
`normalize_adj`表示: $D^{-1/2}AD^{-1/2}$
`preprocess_adj`表示: $I_N+D^{-\frac{1}{2}}AD^{-\frac{1}{2}}\rightarrow \tilde{D}^{-\frac{1}{2}}\tilde{A}\tilde{D}^{-\frac{1}{2}}$
其中:  

​			 $\tilde{A}=A+I_N$ 

​            $\tilde{D}_{ii}=\sum_j\tilde{A}_{ij}$

**参考文献**  
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

1.`laplacian = sp.eye(adj.shape[0]) - adj_normalized`表示:   $L=I_n-D^{-\frac{1}{2}}WD^{-\frac{1}{2}}$
&emsp;&emsp;其中:<a href="https://www.codecogs.com/eqnedit.php?latex=W" target="_blank"><img src="https://latex.codecogs.com/gif.latex?W" title="W" /></a>相当于gcn中的临接矩阵$A$

2.`scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])`表示:  $\qquad\tilde{L}=2L/\lambda_{max}-I_n$

3.`2 * s_lap.dot(t_k_minus_one) - t_k_minus_two`表示:   $\qquad T_k(x)=2xT_{k-1}(x)-T_{k-2}(x)\qquad 其中：T_0=1,T_1=x$

4.

```python
for i in range(2, k+1):
    t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))
```
​	以上表示循环计算$T_k(x)=2xT_{k-1}(x)-T_{k-2}(x)$ 

5.

```python
t_k = list()
t_k.append(sp.eye(adj.shape[0]))
t_k.append(scaled_laplacian)
```
​	以上表示初始化的过程,将scaled Laplacian,令: $\tilde{L}=x$



