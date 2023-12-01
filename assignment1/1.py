#Assumption
#1. Polytope is non-degenerate.
#2. Polytope is bounded
#3. Rak of A is n 


#given feasible point z
import numpy as np

tolerance = 1e-5

def get_rows(A,z,b):
    R = np.dot(A,z)
    tight = []
    untight = []
    for i in range(b.size()):
      if R[i] < tolerance :
        tight.append(i)
      else :
        untight.append(i)
    return tight , untight      


def get_new_z(A2,z,X,b):

    #A[i] * z + alpha * A[i] * X = b[i]
    for i in range(len(A2)):
        difference = b[i] - np.dot(A2[i], z)
        if np.allclose(difference % np.dot(A2[i], X), 0):
           alpha = difference / np.dot(A2[i], X)
           break
    return z + alpha*X


def get_any_vertex(A,b,z):
    if np.dot(A,z) <= b:
        #do nothing
       print("intial point is feasible")
    else:
        return None
    # get tight , untight rows
    tight_rows , untight_rows =  get_rows(A,z,b)
    # find the vertex from intial feasible point
    while(len(tight_rows) < len(z) ):
        A1 = A[tight_rows]
        A2 =  A[untight_rows]
        # get an arbitary vector from nullspace for A1
        null_space_matrix = np.linalg.null_space(A1)
        X = null_space_matrix[:, 0]
        z = get_new_z(A2,z,X,b)
        tight_rows , untight_rows =  get_rows(A,z,b)
    return z


        






        




    
         


    

    
        


