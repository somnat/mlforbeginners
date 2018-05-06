import numpy as np
import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# user-defined input on the dimension and data


class PCA(object):
    @staticmethod
    def cov_mat(X):
        """Objects of this class is a Principal component.

       Examples
       --------
       >>> import numpy as np
       >>> X = np.array([[1,2,3,4,5], [2,3,6,9,0], [8,1,6,4,3]])

       Parameter
       -------------
       X : A numpy array
       """
        row_X, size_row_X = np.shape(X)
        X_mean0 = np.empty(shape=np.shape(X))
        temp_list = list()

        """
         Variables:
         ---------
         row_X: number of features i.e., the number of lists in input matrix X
         size_row_X: number of elements in each each list
         X_mean0 : The transformed X when subtracted from mean
        """
        for i in range(row_X):
            mean_i = (np.sum(X[i]))/size_row_X
            for j in X[i]:
                j = j - mean_i
                temp_list.append(j)
            X_mean0[i] = temp_list
            temp_list = []
        cov_matrix = np.dot(X_mean0, X_mean0.T)/(row_X - 1)
        return cov_matrix

    def principal_component(X):
        Y = PCA.cov_mat(X)
        eigen_val, eigen_vec = np.linalg.eig(Y)
        """ eigenvalue and eigen vectors are extracted through a
         square matrix which is the covariance matrix in our case.
         Variables:
         ---------
         eigen_val: eignevalue of the input data X
         eigen_vec: eignenvector of the input data X
        """
        maximum = np.max(eigen_val)
        index_pc = eigen_val.tolist().index(maximum)
        return eigen_vec[index_pc]
