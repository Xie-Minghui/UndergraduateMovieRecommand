from sklearn import neighbors
import numpy as np
from sklearn import model_selection
import time
import datetime
from sklearn.decomposition import TruncatedSVD
from sklearn.utils.extmath import randomized_svd
from sklearn.neighbors import NearestNeighbors
import math
import copy
'''
函数解释：
    numpy的nonzero函数:
        如果是一维数组，则返回不是0的数的下标。如果是二维数组，则返回元组，第一列是非零元素的行号，第2列是列号
    numpy的argpartition函数：
         k_most_related = k_most_related[np.argpartition(k_most_related,min(k_most_related_r-1,k-1),axis=0)[0:,0]]
         如果找前k小的数，那么相应参数应该是k-1(而不是k),函数返回的是顺序矩阵，类似于nonzero。
         还有一个问题就是如果是numpy的mat矩阵，那么切片之后还是矩阵，所以目前只能访问单个元素，如果要访问一列元素，
         则原矩阵应该转化为numpy的array
'''
def Data_process(file_path:str)->np.mat:
    '''
    传入参数：
        file_path:文件的绝对路径 str
    返回值：
        评分矩阵 numpy.mat，行是用户，列是物品
    '''
    data = np.mat(np.zeros((6100,4100)))  #矩阵行为用户id,列为Movie id
    with open(file_path) as file:
        max_item,max_user = 0,0
        for data_item in file:
            data_item = data_item.strip('\n')
            data_info:list = data_item.split('\t')  #UserID::MovieID::Rating::Timestamp
            # print(data_info)
            data[int(data_info[0])-1,int(data_info[1])-1] = float(data_info[2])
            max_user = max(max_user,int(data_info[0])-1)
            max_item = max(max_item,int(data_info[1])-1)
    return data[:max_user+1,:max_item+1]  #调试数据量

def Mean_centered(origin_data:np.mat)->np.mat:  #
    '''
        传入参数：
            origin_data: 原始未处理的评分矩阵 np.mat
        返回值：
            均值中心化后的数据 numpy.mat
        功能：将矩阵先行均值中心化,即每行非零元素减去非零元素的均值
    '''
    # origin_data = origin_data.T
    r,c = origin_data.shape  #数据矩阵的行数和列数
    mean_centered_data = copy.deepcopy(origin_data) #使用深拷贝，默认的等号是浅拷贝，相当于C++的引用，这是一个陷阱
    for i in range(0,r):
        mean_centered_data[i,np.nonzero(mean_centered_data[i,:])[1]] -= np.mean(mean_centered_data[i,np.nonzero(mean_centered_data[i,:])[1]])
    # for i in range(0,c):
    #     index = np.nonzero(mean_centered_data[:,i])
    #     # r1 = len(index[0])
    #     # print(mean_centered_data[index[0],i].shape)
    #     # print(np.mean(mean_centered_data[index[0],i]))
    #     mean_val = np.mean(mean_centered_data[index[0],i])
    #     for j in index[0]:
    #         mean_centered_data[j,i] -= mean_val
    return mean_centered_data
    # return mean_centered_data.T#和上面origin_data的转置结合起来就是列均值中心化
def Cosine_similarity(inA:np.mat,inB:np.mat,item1:int,item2:int)->float: #调整余弦
    '''
        传入参数：
            inA np.mat 物品的评分向量（SVD降维之后）
            overlap np.array 两个物品被同一个用户评分的用户行号
        返回值：
            相似度 float
        复杂度：O(d) d为降维后的用户维度
    '''
    punishment_factor = [1/math.log(math.e,2+math.fabs(a - b)) for a,b in zip(inA,inB)]
    inA2 = [(a*b)[0,0] for a,b in zip(inA,punishment_factor)]
    inA2 = np.array(inA2)
    inA2 = inA2.reshape(1,inA2.shape[0])
    inA2 = np.mat(inA2)
    numerator = float(inA2 * inB)
    rating_nominator[item1,item2] = rating_nominator[item2,item1] = numerator  #存储分子
    denominatorA[item1,item2] = denominatorA[item2,item1] = np.linalg.norm(inA)
    denominatorB[item1,item2] = denominatorB[item2,item1]= np.linalg.norm(inB)
    denominator =  denominatorA[item1,item2] * denominatorB[item1,item2] #乘法改成加法
    rating_denominator[item1,item2] = rating_denominator[item2,item1] = denominator ** 2 #存储分母两个元素
    return numerator/denominator

def Pearson_similarity(inA:np.mat,inB:np.mat)->float:
    '''
    传入参数：
            inA np.mat 物品的评分向量（SVD降维之后）
            overlap np.array 两个物品被同一个用户评分的用户行号
        返回值：
            相似度 float
        复杂度：O(d) d为降维后的用户维度
    '''
    return np.corrcoef(inA,inB,rowvar = 0)[0][1]



def Choose_dimension(Sigma:np.array,info_remain_percent:float)->int:
    '''
        传入参数：
            Sigma:奇异值向量 np.array
            info_remain_percent:降维后的矩阵应该保留原矩阵的信息占比
        返回值；
            选取前ans个奇异值即可保留原矩阵信息的info_remain_percent,即前i个奇异值和占所有奇异值总和的比率>=info_remain_percent  int
        复杂度:O(n),n表示列的个数
    '''
    totle_sum = sum(Sigma**2)
    sum_now = 0.0
    ans = 0
    Len = len(Sigma)  #求Sigma向量的长度
    for i in range(0,Len):
        sum_now += Sigma[i]**2
        if sum_now/totle_sum >= info_remain_percent:
            ans = i
            break
    return ans + 1
def Calculate_items(origin_data:np.mat, mean_centered_data_transf:np.mat,Similarity_calculate_means)->np.mat:
    '''
        传入参数：mean_centered_data_transf 降维后的矩阵 np.mat
                similarity_calculate_means 相似度计算方法
        返回值：similarity_item_matrix 物品之间的相似度矩阵 np.mat
        复杂度：O(d*c1^2),d表示用户行降维后的个数，c1表示用户所评分的最大电影数，相比较，朴素算法O(c^2*d),
        第一种比第二种优，因为一个物品被评分的用户数一般>>一个用户评分的物品数
    '''
    mean_centered_data_transf = mean_centered_data_transf.T #将降维后的矩阵转置，使之符合原始矩阵
    r,c = mean_centered_data_transf.shape  #分别是降维后的列数（用户维度）和 物品数
   
    similarity_item_matrix = np.mat(np.zeros((c,c)))
    print("c: %d"%c)
    for i in range(0,c):
        print(i)
        for j in range(i+1,c):
            # print(i,j)
            # overlap = np.nonzero(np.logical_and(origin_data[:,i].A>0,origin_data[:,j]>0))[0]
            # if i == 1 and j == 4:
            #     print('nihao',overlap)
            similarity_item_matrix[i,j] = similarity_item_matrix[j,i] = Similarity_calculate_means(mean_centered_data[:,i],mean_centered_data[:,j],i,j)
    return similarity_item_matrix

def Calculate_items_similarty(origin_data:np.mat,mean_centered_data:np.mat)->np.mat:
    '''
        传入参数：
            mean_centered_data:均值中心化后的矩阵 np.mat
        返回值；
            item之间的相似度矩阵
    '''
    U,Sigma,VT = np.linalg.svd(mean_centered_data)  #numpy中SVD的实现
    dimension = Choose_dimension(Sigma,0.90) #计算应该降维的维度
    print("用户维度降维：",dimension)
    '''
    #sklearn的截断SVD实现
    svd = TruncatedSVD(n_components = dimension)
    mean_centered_data_transf = svd.fit_transform(mean_centered_data) #降维后的评分矩阵
    Sigma = svd.singular_values_
    #sklearn的随机SVD实现(近似解，相对于截断SVD速度更快)
    #截断SVD使用精确解算器ARPACK，随机SVD使用近似技术。
    U,Sigma,VT = randomized_svd(mean_centered_data,n_components = dimension)
    
    '''
    Sigma_dimension = np.mat(np.eye(dimension)*Sigma[:dimension])  #将奇异值向量转换为奇异值矩阵
    # print(Sigma_dimension)
    # print('nihao:',Sigma)
    # print(U[:,:dimension])
    mean_centered_data_transf = mean_centered_data.T*U[:,:dimension]*Sigma_dimension.I #降维后的评分矩阵
    # print(mean_centered_data_transf)
    # print("奇异值分解，且降维后的矩阵：\n",mean_centered_data_transf.T)
    print("nihao1")
    similarity_item_matrix = Calculate_items(origin_data,mean_centered_data_transf,Cosine_similarity)#Pearson_similarity
    return similarity_item_matrix
def ItemRecommend2(origin_data:np.mat,mean_centered_data:np.mat, metric:str,user:int,k:int,predict_num:int,recommend_reasons_num:int)->np.array:
    U,Sigma,VT = np.linalg.svd(mean_centered_data)  #numpy中SVD的实现
    dimension = Choose_dimension(Sigma,0.90) #计算应该降维的维度
    Sigma_dimension = np.mat(np.eye(dimension)*Sigma[:dimension])  #将奇异值向量转换为奇异值矩阵
    mean_centered_data_transf = mean_centered_data.T*U[:,:dimension]*Sigma_dimension.I #降维后的评分矩阵
    # print(mean_centered_data_transf.shape)
    mean_centered_data_transf = mean_centered_data_transf.T
    # print(mean_centered_data_transf.shape)
    # mean_centered_data_transf = mean_centered_data_transf.T
    # mean_centered_data_transf = np.array(mean_centered_data_transf)
    user -= 1
    rated_item = np.nonzero(origin_data[user,:]>0)[1] #获得目标用户已评分的物品的标号
    # len_rated_item = len(rated_item)
    # for i in range(0,len_rated_item):
        # origin_rated_item[i] = rated_item 
    # print(rated_item.shape)
    unrated_item = np.nonzero(origin_data[user,:]<0.5)[1] #获得目标用户未评分的物品的标号
    # print('ra',rated_item)
    # print('un',unrated_item)
    len_unrated_item = len(unrated_item)
    predict_rating = np.array(np.zeros((len_unrated_item+5,2+recommend_reasons_num)))
    for i in range(0,len_unrated_item):
        model_knn =  NearestNeighbors( algorithm = 'ball_tree')
        # print("oh")
        # print(mean_centered_data_transf[:,rated_item].shape)
        model_knn.fit(mean_centered_data_transf[:,rated_item].T) 

        # print(mean_centered_data_transf[:,rated_item].T.shape)
        # print("ni",rated_item.shape)
        # k = min(k,len(rated_item))
        # print(k,len(rated_item))
        # print(mean_centered_data_transf[:,unrated_item[i]].shape)
        # k = 2
        distances, indices = model_knn.kneighbors(mean_centered_data_transf[:,unrated_item[i]].T #必须是行向量
            , n_neighbors = k)
        similarities = distances.flatten() - 1 
        sum_similarities = sum(similarities)
        # print(similarities)
        # print('indices',indices)

        k_most_related = np.array(np.zeros((k,2)))
        for j in range(0,k):
            predict_rating[i][0] += similarities[j]*origin_data[user,rated_item[indices.flatten()[j]]]
            k_most_related[j][0] = similarities[j]
            k_most_related[j][1] = rated_item[indices[0][j]]  #不是直接使用indices[0][j]
        if math.fabs(sum_similarities - 0) < 1e-2:
            predict_rating[i][0] = -1
        else:
            # print('nihao',sum_similarities)
            # print(predict_rating[i][0])
            predict_rating[i][0] /= sum_similarities
            predict_rating[i][1] = unrated_item[i]
       
        # print(predict_rating[i][0])
        # k_most_related = k_most_related[np.argpartition(k_most_related,min(recommend_reasons_num-1,k-1),axis=0)[0:,0]] #小的放前面，min(k_most_related_r-1,k-1),因为已评分的物品小于k个
        # print('k_most_related',k_most_related)
        recommend_reasons_items = k_most_related[np.argpartition(k_most_related,min(recommend_reasons_num-1,k-1),axis = 0)[0:min(recommend_reasons_num,k),0]][:,1]
        predict_rating[i][2:min(2+recommend_reasons_num,2+k)] = recommend_reasons_items
        # print("nieshi")
    predict_rating = predict_rating[np.argpartition(predict_rating[0:len_unrated_item],min(len_unrated_item-1,predict_num-1),axis=0)[0:min(len_unrated_item,predict_num),0]]
    predict_rating = predict_rating[predict_rating[:,0].argsort()]
    # print(predict_rating)
    predict_rating = predict_rating[::-1,:]
    r2,c2 = predict_rating.shape

    score = predict_rating[:,0]
    pos = predict_rating[:,1]
    recommend_reasons_items_final = predict_rating[:,2:min(2+recommend_reasons_num,2+k)] #最优物品编号
    score,pos = np.array(score),np.array(pos)
    top_k_item = pos[0:predict_num] #评分最高的物品在unrated_item中的位置
    top_k_score = score[0:predict_num]
    return top_k_item,top_k_score,recommend_reasons_items_final

def ItemRecommend(origin_data:np.mat,similarity_item_matrix:np.mat, user:int,k:int,predict_num:int)->np.array:
    '''
        传入参数：similarity_item_matrix np.mat 物品相似度矩阵
                data np.mat  原始评分矩阵
                user int 要推荐的用户的id
                k int 预测未评分物品分数时，参考的最近邻的个数
                predict_num int 推荐电影的个数 
        返回值：   unrated[top_k]   推荐物品的id
        复杂度：  O(n1*n2) n1表示已评分的个数，n2表示未评分的个数
    '''
    user -= 1 #user从0开始
    r,c = origin_data.shape

    rated_item = np.nonzero(origin_data[user,:]>0)[1] #获得目标用户已评分的物品的标号
    # print('ra',rated_item)
    unrated_item = np.nonzero(origin_data[user,:]<0.5)[1] #获得目标用户未评分的物品的标号
    # print('un',unrated_item)
    len1 = len(rated_item)
    len2 = len(unrated_item)
    print(len1,len2)
    # print(similarity_item_matrix)

    recommend_reasons_num = 2
    #因为实际情况必然是对当前预测正效应的物品较多(>=3),所以不用特意去筛选负效应的物品，因为负效应的物品（或者是正效应很小的物品）不应该出现在推荐理由上
    predict_rating = np.array(np.ones((len2+5,2+recommend_reasons_num))) #第一列存储相似度，第二列存储物品的标号,后面(5-2)存储5-2个推荐理由
    #predict_ratiing创建初始化为0，可能初始值0与推荐物品编号冲突，但并不会，因为后面坐了切片，后面无效的0都被舍去了
    for i in range(0,len2):  #未评分的物品
        k_most_related = np.array(np.zeros((len1+5,2)))  #第i个未评分的物品与所有已评分物品的相似度，然后筛选出k近邻
        for j in range(0,len1): #评分的物品
            k_most_related[j,0] = -similarity_item_matrix[rated_item[j],unrated_item[i]]   #当前第i个未评分的物品与第j个评分物品的相似度
            k_most_related[j,1] = rated_item[j]  #相似度，添加符号转化为k小值，存储在原数据矩阵的下标
        '''
            numpy argpartition 返回每一列的前K个值的位置(axis=0时)，是一个二维矩阵
        '''

       
        k_most_related = k_most_related[0:len1] #截取前面有效的信息，已评分物品的数量
        # print(k_most_related)
        # print(k_most_related.shape)
        k_most_related_r,k_most_related_c = k_most_related.shape
        # print(k_most_related)
        k_most_related = k_most_related[np.argpartition(k_most_related,min(k_most_related_r-1,k-1),axis=0)[0:,0]] #小的放前面，min(k_most_related_r-1,k-1),因为已评分的物品小于k个
        # print('nihao1',k_most_related[:,0])
        # print(sum(k_most_related[:,0]))
        # print(origin_data[user,(k_most_related[:,1]).astype(np.int32)])
        # predict_score = np.inner(k_most_related[:,0] , np.array((origin_data[user,(k_most_related[:,1]).astype(np.int32)])))/sum(np.fabs(k_most_related[:,0]))
        numerator = 0.0
        denominator = 0.0
        # print(k_most_related[np.argpartition(k_most_related,min(recommend_reasons_num-1,k_most_related_r-1),axis = 0)[0:,0]])
        recommend_reasons_items = k_most_related[np.argpartition(k_most_related,min(recommend_reasons_num-1,k_most_related_r-1),axis = 0)[0:min(recommend_reasons_num,k_most_related_r),0]][:,1]
        for v in k_most_related:
            if v[0] < 0:  #只计算正效应（相似度为正的物品），前面为了取前k大，每个相似度添加了负号，转化为前k小
                numerator += v[0]*origin_data[user,int(v[1])]
                denominator += np.fabs(v[0])
        # if i == 1 or i == 5:
        #     print(k_most_related)
        #     print(numerator,denominator,"jifdjfio")
        if np.fabs(denominator - 0.0) <= 1e-2:
            predict_score = 0
        else:
            predict_score = numerator/denominator  
            '''
                如果不除以分母，预测评分偏差会比较大（偏大），如果除以分母,考虑一种极端情况，如果一个用户的所有评分都是一样的
                那么预测评分都会是这个相同的分数
            '''
        predict_rating[i,0] = predict_score #第i个未评分物品的预测分数
        # print(predict_score)
        predict_rating[i,1] = unrated_item[i] #相应的物品编号
        # print(recommend_reasons_items.shape)
        # print(recommend_reasons_items)
        predict_rating[i,2:min(2+recommend_reasons_num,2+k_most_related_r)] = recommend_reasons_items  #存储推荐理由物品的编号，因为你看过...
    
    # item  = np.argpartition(predict_rating[:],predict_num,axis = 0)[0:k,:]
    # print(predict_rating)

    #[0:min(len2,predict_num),0],不是[0:min(len2,predict_num)-1,0]，因为区间是左闭右开
    predict_rating = predict_rating[np.argpartition(predict_rating[0:len2],min(len2-1,predict_num-1),axis=0)[0:min(len2,predict_num),0]]
    #上面的predict_rating[0:len2]切片舍弃了无效信息

    # print('fddfdsadf:\n',predict_rating)
    # predict_rating = np.sort(predict_rating[0,:],axis = 0)
    predict_rating = predict_rating[predict_rating[:,0].argsort()]
    # print(predict_rating)
    # pos = predict_rating[:-1,1].astype(np.int32)
    # print(pos)
    # print(origin_data[:,pos[0]])
    # print(pos)
    r2,c2 = predict_rating.shape
    # print(predict_rating)
    # score,pos = [],[]
    # for i in range(r2-1,-1,-1):
    #     score.append(predict_rating[i][0])
    #     pos.append(predict_rating[i][1])
        # print(predict_rating[i][0],predict_rating[i][1])
    # print(predict_rating)
    score = -predict_rating[:,0]
    pos = predict_rating[:,1]
    recommend_reasons_items_final = predict_rating[:,2:min(2+recommend_reasons_num,2+k_most_related_r)] #最优物品编号
    # print(predict_rating.shape)
    # print(pos,score)
    # score = predict_rating[k-predict_num:-1,0]
    # print('nimei:',score)
    # print(pos)
    score,pos = np.array(score),np.array(pos)
    top_k_item = pos[0:predict_num] #评分最高的物品在unrated_item中的位置
    top_k_score = score[0:predict_num]
    return top_k_item,top_k_score,recommend_reasons_items_final
    
# def main_test():
start = time.clock()
file_path = 'E:\Bigdata\ml-100k\\u.data'
origin_data = Data_process(file_path)
# print(origin_data)
origin_data[196-1,242-1] = 0.0
origin_data[186-1,302-1] = 0.0
rating_denominator = np.zeros((origin_data.shape[1],origin_data.shape[1])) #相似度计算分母矩阵
denominatorA = np.mat(np.zeros((origin_data.shape[1],origin_data.shape[1])))
denominatorB = np.mat(np.zeros((origin_data.shape[1],origin_data.shape[1])))
rating_nominator = np.zeros((origin_data.shape[1],origin_data.shape[1])) #相似度计算分子矩阵

user = 3
mean_centered_data = Mean_centered(origin_data)
# print(mean_centered_data)  #打印均值中心化的评分矩阵
similarity_item_matrix=Calculate_items_similarty(origin_data,mean_centered_data)
# print(similarity_item_matrix)  #打印相似度矩阵
print('nihao0')
# r,c = similarity_item_matrix.shape
# # print(r,c)
# for i in range(0,c):
#     for j in range(0,c):
#         print(i,j,similarity_item_matrix[i,j])
top_k_item,top_k_score,recommend_reasons_items = ItemRecommend(origin_data,similarity_item_matrix,196,2,2)
# top_k_item,top_k_score,recommend_reasons_items = ItemRecommend2(origin_data,mean_centered_data,'minkowski',3,2,2,2)

for item,score,reason_items in zip(top_k_item,top_k_score,recommend_reasons_items):
    print()
    print("推荐的电影：%d\n预测用户 %d 对电影 %d 的评分为：%f"%(item+1,user,item,score))
    print("因为用户%d之前看过"%user,end = ' ')#
    for it in reason_items:
        print("电影%d"%it,end = ' ')
    print()
    # print(reason_items)
    # print("推荐的电影：{},预测 {} 对电影 {} 的评分为：{}",item,user,item,score,)
    # print("因为用户%d之前看过电影",user,end = ' ')
    # print(reason_items)
# print(top_k_item)
# print("相应的评分：")
# print(top_k_score)
# print("相应的推荐理由")
# print(recommend_reasons_items)  #推荐理由也可以了
# print('nihao')
while(True):
    print("please input user item rating or -1 -1 -1 to exit!")
    user,item,rating = map(int,input().split(" "))
    user -= 1
    item -= 1
    origin_data[user,item] = rating
    user_item = np.nonzero(origin_data[user,:])[1]
    user_av = np.mean(origin_data[user,user_item])
    if(user < 0):
        break
    for item1 in user_item:
        # print(item1)
        if(item1 == item):
            continue
        else:
            r0 = origin_data[user,item]
            r1 = origin_data[user,item1]
            rating_nominator[item1,item] = rating_nominator[item,item1] = (rating_nominator[item1,item] + (r0 - user_av) * (r1 - user_av) * (1/math.log(math.e,2+math.fabs(r0 - r1))))
            # print(type(denominatorA))
            denominatorA[item,item1] = denominatorA[item1,item] = (denominatorA[item1,item] + (r0 - user_av) ** 2)
            denominatorB[item,item1] = denominatorB[item1,item] = (denominatorB[item1,item] + (r1 - user_av) ** 2)
            rating_denominator[item,item1] = rating_denominator[item1,item] = (denominatorA[item,item1] * denominatorB[item1,item])
            similarity_item_matrix[item,item1] = similarity_item_matrix[item1,item] =rating_nominator[item1,item]/ math.sqrt(rating_denominator[item,item1])
    top_k_item,top_k_score,recommend_reasons_items = ItemRecommend(origin_data,similarity_item_matrix,3,2,2)
# top_k_item,top_k_score,recommend_reasons_items = ItemRecommend2(origin_data,mean_centered_data,'minkowski',3,2,2,2)
    for item,score,reason_items in zip(top_k_item,top_k_score,recommend_reasons_items):
        print() 
        print("推荐的电影：%d\n预测用户 %d 对电影 %d 的评分为：%f"%(item+1,user,item,score))
        print("因为用户%d之前看过"%user,end = ' ')#
        for it in reason_items:
            print("电影%d"%it,end = ' ')
        print()

end = time.clock()

print(end - start)


# main_test()

'''
    工作日志，10.31下午添加了给出推荐理由的功能，推荐你i物品，因为你看过j物品
    1.12 添加了惩罚因子（调整余弦）
'''
