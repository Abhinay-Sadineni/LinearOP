#Assumption
#1. Polytope is non-degenerate.
#2. Rank of A is n 


from scipy.linalg import null_space
#given feasible point z
import numpy as np

tolerance = 1e-5

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


def get_new_z(A,A2,z,X,b):

    #A[i] * z + alpha * A[i] * X = b[i]
    for i in range(len(A2)):
        difference = b[i] - np.dot(A2[i], z)
        if not np.allclose(np.dot(A2[i], X), 0,rtol= tolerance):
            alpha = difference / np.dot(A2[i], X)
            if(alpha != 0 ) :
                if np.all(np.dot(A,z+alpha*X)-b <= tolerance):
                    return z + alpha*X, True
    return z, False  
    
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
            z,flg = get_new_z(A,A2,z,X,b)
            if not flg :
               print("Not Possible\n")
               break
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
            z,flg = get_new_z(A,A2,z,X,b)
            if not flg :
               print("Not Possible\n")
               break
            tight_rows , untight_rows =  get_rows(A,z,b)

    return z

def get_opt_vertex(A,z,C,b):
    tight_rows , untight_rows =  get_rows(A,z,b)
    A1 = np.array([A[i] for i in tight_rows])
    # print(A1)
    A2 =  [A[i] for i in untight_rows]
    coeff = np.linalg.lstsq(A1.T, C, rcond=None)[0]
    while not np.all(coeff > 0) :
       i = np.where(coeff < 0)[0][0]
       A1_inv = np.linalg.inv(A1)
       c = A1_inv[:,i]
       c = -1*c.flatten()
       z,flg = get_new_z(A,A2,z,c,b)

       if not flg :
               print("Unbounded\n")
               break
       tight_rows , untight_rows =  get_rows(A,z,b)
       A1 = np.array([A[i] for i in tight_rows])
       A2 =  [A[i] for i in untight_rows]
       coeff = np.linalg.lstsq(A1.T, C, rcond=None)[0]
    return z
    



def main():
    # arr = np.loadtxt("sample_data.csv", delimiter=",", dtype=float)

    # # Extracting z, A, c, and b from the loaded data
    # z = arr[0, :-1]  # Initial feasible point, excluding the last element
    # c = arr[1, :-1]  # Cost vector, excluding the last element
    # b = arr[2:, -1]  # Constraint vector, last column excluding the top two elements
    # A = arr[2:, :-1]  # Matrix A, excluding the last column and top two rows

    # # Now you have z, A, c, and b
    # print("z:", z)
    # print("A:", A)
    # print("c:", c)
    # print("b:", b)

    # #get optimum vertex
    # Z = get_opt_vertex(A,z,c,b)
    A = [[1,1],[-1,-1],[-1,0],[0,-1]]
    b = [2,-1,0,0]
    C = [1,0.5]
    z = [0.7,0.3]
    z = get_any_vertex(A,b,z)
    print(z)
    z = get_opt_vertex(A,z,C,b)
    print(z)
    print(z)

if __name__ == "__main__":
    main()






        




    
         


    

    
        


