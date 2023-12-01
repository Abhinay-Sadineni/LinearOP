#Assumption
#1. Polytope is non-degenerate.
#2. Polytope is bounded
#3. Rak of A is n 


#given feasible point z
import np as numpy

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

def get_any_vertex(A,b,c,z):
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
        nullspace = np.linalg.null_space(A1)
        X = null_space_matrix[:, 0]
        


        




    
         


    

    
        


