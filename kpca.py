import tensorflow as tf

class KPCA:
    def __init__(self, n_components, gamma = None) -> None:
        self.n_components = n_components
        self.gamma = gamma
        self.fitted = False
        self.train_data = None
    
    @staticmethod
    def __rbf_kernel(input_, train_data, gamma):
        # Calculating the squared Euclidean distances for every pair of points
        # in the MxN dimensional dataset.
        # Converting the pairwise distances into a symmetric MxM matrix.
        distance_matrix =  - 2 * tf.matmul(input_, tf.transpose(train_data)) + tf.reduce_sum(train_data**2, axis=1) + tf.reduce_sum(input_**2, axis=1)[:, tf.newaxis]

        # Computing the MxM kernel matrix.
        return tf.exp(-gamma * distance_matrix)
    
    def fit(self, input_):
        self.fitted = True
        self.train_data = input_
        if self.gamma == None:
            self.gamma = 1 / input_.shape[-1]
            
        k = KPCA.__rbf_kernel(input_, input_, self.gamma)
        
        # Centering the symmetric NxN kernel matrix.
        self.N = k.shape[0]
        ones = tf.ones([self.N,self.N], dtype=tf.float64)/self.N
        self.center_k = k - tf.matmul(ones, tf.transpose(k)) - tf.matmul(k, tf.transpose(ones)) + tf.matmul(tf.matmul(ones, tf.transpose(k)), tf.transpose(ones))
        # Obtaining eigenvalues in descending order with corresponding
        # eigenvectors from the symmetric matrix.
        eigval, eigvec = tf.linalg.eigh(self.center_k)
        # Obtaining the i eigenvectors that corresponds to the i highest eigenvalues.
        self.trans_eigvec = tf.stack([eigvec[:,-i] for i in range(1,self.n_components+1)], axis=1)
        self.trans_eigval = eigval[-self.n_components:]
    
    def fit_transform(self, input_):
        if (input_ != self.train_data).all():
            self.fit(input_)
        return tf.matmul(self.center_k, tf.math.divide(self.trans_eigvec, tf.math.sqrt(self.trans_eigval)))
    
    def transform(self, input_):
        try: 
            if self.fitted == False:
                raise Exception('Need to call fit method first!')
            if input_ == self.train_data:
                return tf.matmul(self.center_k, tf.math.divide(self.trans_eigvec, tf.math.sqrt(self.trans_eigval)))
            
            k = KPCA.__rbf_kernel(input_, self.train_data, self.gamma)
            Nx = k.shape[0]
            ones_n = tf.ones((Nx, self.N), dtype=tf.float64)/self.N
            ones_old = tf.ones((self.N, self.N), dtype=tf.float64)/self.N
            k = k - tf.matmul(ones_n, self.center_k) - tf.matmul(k, ones_old) + tf.matmul(tf.matmul(ones_n, self.center_k), ones_old)
            return tf.matmul(k, tf.math.divide(self.trans_eigvec, tf.math.sqrt(self.trans_eigval)))
        
        except Exception as e:
            print(str(e))