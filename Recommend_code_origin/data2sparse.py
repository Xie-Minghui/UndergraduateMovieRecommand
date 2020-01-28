from scipy.sparse import csr_matrix
from scipy.sparse import save_npz
import numpy as np
import os

def ReadData(file_path:str) -> np.array:
    #从文件中读取文件并且返回scipy矩阵需要的行，列，值3个array
    row,col,data = [],[],[]
    with open(file_path) as file:
        for fileline in file:
            fileline = fileline.strip('\n')
            fileline_list = fileline.split('\t') #读取的是str，要转化为float，然后再转化为array
            row.append(float(fileline_list[0])),col.append(float(fileline_list[1])),data.append(float(fileline_list[2]))
    return np.array(row),np.array(col),np.array(data)

file_path = os.path.join(os.getcwd(),'ml-100k//u.data')
row,col,data = ReadData(file_path)
sparse_matrix = csr_matrix((data,(row,col)),shape = (2000,2000)) #建立稀疏矩阵
save_npz('sparse_matrix_100k',sparse_matrix) #稀疏矩阵压缩存储到npz文件