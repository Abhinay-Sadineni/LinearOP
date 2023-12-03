#team details
#Sadineni Abhinay - CS21BTECH11055
#G harshavardhan reddy -CS21BTECH11017
#Karthik Kotikalapudi - CS21BTECH11030
#Akkasani yagnesh reddy - CS21BTECH11003




#Assumption
#1. Rank of A is n 
#2. feasibile point should be generated


from itertools import combinations
from scipy.linalg import null_space
#given feasible point z
import numpy as np

tolerance = 1e-5
np.random.seed(42)

def get_rows(A,z,b):
    R = np.dot(A,z)-b
    tight = []
    untight = []
    for i in range(len(b)):
      if abs(R[i]) < tolerance :
        tight.append(i)
      else :
        untight.append(i)
    return tight , untight      

def solve(A,b):
    try:
     x = np.linalg.solve(A,b)
     return x
    except np.linalg.LinAlgError:
     return None

def get_vertex_point_from_comb(A,b):
    indices = [i for i in range(len(A))]
    all_forms = combinations(indices ,len(A[0]))
    for i in all_forms:
          result = solve([A[idx] for idx in i ],[b[idx] for idx in i ])
          if result is None: continue
          else: 
            if(np.all(np.dot(A,result)<=b)):return result
    return None    



from scipy.optimize import root_scalar

tolerance = 1e-7
global b
def get_rows(A,z,b):
    R = np.dot(A,z)-b
    tight = []
    untight = []
    for i in range(len(b)):
      if abs(R[i]) < tolerance :
        tight.append(i)
      else :
        untight.append(i)
    return tight , untight      


# def get_new_z(A,A2,z,X,b):
#     #A[i] * z + alpha * A[i] * X = b[i]
#     for i in range(len(A2)):
#         difference = b[i] - np.dot(A2[i], z)
#         if not np.isclose(np.dot(A2[i], X), 0,rtol= tolerance):
#             def equation(alpha):
#                 return np.dot(A[i], (z + alpha * X)) - b[i]
#             result = root_scalar(equation, x0=0.0, method='newton', xtol=tolerance)
#             # alpha = difference / np.dot(A2[i], X)
#             alpha = result.root
#             if(alpha != 0 ) :
#                 if np.all(np.dot(A,z+alpha*X)-b <= tolerance):
#                     return z + alpha*X, True
#     return z, False 

def get_alpha(A,z,X,i,b):
    if not np.allclose(np.dot(A[i], X), 0,rtol= tolerance):
        def equation(alpha):
            return np.dot(A[i], (z + alpha * X)) - b[i]
        result = root_scalar(equation, x0=0.0, method='newton', xtol=tolerance)
        return result.root
    else :
        return 0  

def isvalid(A,z,u,i,b):
    alpha = get_alpha(A,z,u,i,b)
    if(alpha != 0) : 
        z_prime = z + alpha * u
        if(np.all(np.matmul(A,z_prime)<=b)): 
            return True
    else : 
        return False 
    
def get_any_vertex(A,b,z):
    if np.all(np.dot(A,z) <= b):
        #do nothing
       print("intial point is feasible")
    else:
        return None
    # get tight , untight rows
    tight_rows , untight_rows =  get_rows(A,z,b)
    # find the vertex from intial feasible point
    while(len(tight_rows) < len(z)):
        if(len(tight_rows) == 0):
            X = np.ones(len(z))
            A1 = [A[i] for i in tight_rows]
            A2 =  [A[i] for i in untight_rows]
            # z,flg = get_new_z(A,A2,z,X,b)
            # if flg ==False :
            #    print("Not Possible\n")
            #    break
            alpha= 0
            for i in untight_rows:
                if(isvalid(A,z,X,i,b)) :
                    alpha = get_alpha(A,z,X,i,b)
            z = z +alpha*X
            tight_rows , untight_rows =  get_rows(A,z,b)
        else :
            # print(tight_rows)
            A1 = [A[i] for i in tight_rows]
            A2 =  [A[i] for i in untight_rows]
            # get an arbitary vector from nullspace for A1
            null_space_matrix = null_space(A1)
            # print(A1,"   ",null_space_matrix)
            # print(null_space_matrix)
            X = null_space_matrix[:, 0]
            # z,flg = get_new_z(A,A2,z,X,b)
            # if not flg :
            #    print("Not Possible\n")
            #    break
            alpha= 0
            for i in untight_rows:
                if(isvalid(A,z,X,i,b)) :
                    alpha = get_alpha(A,z,X,i,b)
            z = z +alpha*X
            tight_rows , untight_rows =  get_rows(A,z,b)

    return z

def get_opt_vertex(A,z,C,b):
    tight_rows , untight_rows =  get_rows(A,z,b)
    A1 =A[tight_rows]
    # print(A1)
    A2 =  [A[i] for i in untight_rows]
    coeff = np.linalg.lstsq(A1.T, C, rcond=None)[0].flatten()
    while not np.all(coeff > -tolerance):
       i = np.where(coeff < -tolerance)[0][0]
       print("i: ",i)

       print(len(A1))
       print(len(A1[0]))
       if len(A1) > len(A1[0]):
            print("Degenerate case\n")
            epsilon = 1e-5
            b = remove_degenerate(A,b,z)
            print(np.dot(A,z).shape,b.shape)
            z = get_any_vertex(A,b,z)
            print("gg")
            z = get_opt_vertex(A,z,C,b)
            # print("z: ",z)
            break 
       A1_inv = np.linalg.inv(A1)
       c = A1_inv[:,i].flatten()
    
       c = -1*c
    #    z,flg = get_new_z(A,A2,z,c,b)
       alpha= 0
       flg = False
       for i in untight_rows:
            print("y")
            if(isvalid(A,z,c,i,b)) :
                flg = True
                alpha = get_alpha(A,z,c,i,b)
            print("y")
       if(flg == False) :
           print("Unbounded")
           return
       z = z +alpha*c

    #    if not flg :
    #            print("Not Possible\n")
    #            break
       initial_size = len(A1)
       tight_rows , untight_rows =  get_rows(A,z,b)
       print("nbnsbd")
       A1 = np.array([A[i] for i in tight_rows])
       A2 =  [A[i] for i in untight_rows]
       new_size = len(A1)
       print(new_size)
       print(initial_size)
    #    if (new_size-initial_size > 1):
    #         print("Degenerate case\n")
    #         epsilon = 1e-5
    #         #adding infinitesimally small epsilon to b vector, perturbing b vector
    #         z = get_opt_vertex(A,z,C,b)
    #         print("z: ",z)
    #         break
       print("old coeff:", coeff)
       coeff = np.linalg.lstsq(A1.T, C, rcond=None)[0]
       print("coeff:", coeff)
    return z
    
def remove_degenerate(A,b,z):
    rows = A.shape[0]-A.shape[1]
    count = 1
    while 1:
        if count < 1000:
            count += 1
            B = b.copy()
            B[:rows] += np.random.uniform(1e-6,1e-5,size=rows)
        else:
            B = b.copy()
            B[:rows] += np.random.uniform(0.1,10,size=rows)
        tight_rows , untight_rows =  get_rows(A,z,B)
        A1 =A[tight_rows]
        print("remove:",len(A1))
        print("remove:",len(A1[0]))
        print(B)
        if len(A1) <= len(A1[0]):
            print("Degeneracy removed")
            break
    return B
        


# A = [[-1,0,0],[0,-1,0],[0,0,-1],[1,1,1]]
# b = [0,0,0,8]
# C = [0,0,0]
# z = [2,1,3]
# z = get_any_vertex(A,b,z)
# print(z)
# z = get_opt_vertex(A,z,C,b)
# print(z)

def main():
    arr = np.loadtxt("test/test_cases_4/2.csv", delimiter=",", dtype=float)

    # Extracting z, A, c, and b from the loaded data
     # Initial feasible point, excluding the last element
    c = arr[0, :-1]  # Cost vector, excluding the last element
    b = arr[1:, -1]  # Constraint vector, last column excluding the top two elements
    A = arr[1:, :-1]  # Matrix A, excluding the last column and top two rows
    z = get_vertex_point_from_comb(A,b)
    # Now you have z, A, c, and b
    print("z:", z)
    print("A:", A)
    print("c:", c)
    print("b:", b)

    z = get_any_vertex(A,b,z)
    print(z)

    #get optimum vertex
    z = get_opt_vertex(A,z,c,b)
    print(z)

if __name__ == "__main__":
    main()
