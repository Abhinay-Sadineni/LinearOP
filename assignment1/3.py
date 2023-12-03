#Assumption
#3. Rank of A is n 

from scipy.linalg import null_space
from scipy.optimize import root_scalar
#given feasible point z
import numpy as np

tolerance = 1e-5
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

def get_alpha(A,z,X,i,b):
    if not np.allclose(np.dot(A[i], X), 0,rtol= tolerance):
        def equation(alpha):
            return np.dot(A[i], (z + alpha * X)) - b[i]
        def derivative(alpha):
            return np.dot(A[i], X)
        result = root_scalar(equation, x0=0.0, method='newton', fprime=derivative, xtol=tolerance)
        return result.root 

def isvalid(A, z, u, i, b):
    alpha = get_alpha(A, z, u, i, b)
    if alpha is not None and alpha != 0:
        z_prime = z + alpha*u
        if np.all(np.matmul(A, z_prime) <= b):
            return True
    return False
    
def get_any_vertex(A,b,z):
    if np.all(np.dot(A,z) <= b):
       print("intial point is feasible")
    else:
        return None
    tight_rows , untight_rows =  get_rows(A,z,b)
    while(len(tight_rows) < len(z)):
        if(len(tight_rows) == 0):
            X = np.ones(len(z))
            A1 = [A[i] for i in tight_rows]
            A2 =  [A[i] for i in untight_rows]
            alpha= 0
            for i in untight_rows:
                if(isvalid(A,z,X,i,b)) :
                    alpha = get_alpha(A,z,X,i,b)
            z = z +alpha*X
            tight_rows , untight_rows =  get_rows(A,z,b)
        else :
            A1 = [A[i] for i in tight_rows]
            A2 =  [A[i] for i in untight_rows]
            null_space_matrix = null_space(A1)
            X = null_space_matrix[:, 0]
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
    A2 =  [A[i] for i in untight_rows]
    coeff = np.linalg.lstsq(A1.T, C, rcond=None)[0].flatten()
    while not np.all(coeff > -tolerance):
       i = np.where(coeff < -tolerance)[0][0]
       if len(A1) > len(A1[0]):
            b = remove_degenerate(A,b,z)
            z = get_any_vertex(A,b,z)
            z = get_opt_vertex(A,z,C,b)
            break 
       A1_inv = np.linalg.inv(A1)
       c = A1_inv[:,i].flatten()
       c = -1*c
       alpha= 0
       flg = False
       for i in untight_rows:
            if(isvalid(A,z,c,i,b)) :
                flg = True
                alpha = get_alpha(A,z,c,i,b)
       if(flg == False) :
           print("Unbounded")
           return
       z = z +alpha*c
       initial_size = len(A1)
       tight_rows , untight_rows =  get_rows(A,z,b)
       A1 = np.array([A[i] for i in tight_rows])
       A2 =  [A[i] for i in untight_rows]
       new_size = len(A1)
       coeff = np.linalg.lstsq(A1.T, C, rcond=None)[0]
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
        if len(A1) <= len(A1[0]):
            break
    return B

def main():
    myfile = input("Enter file name: ")
    arr = np.loadtxt(myfile, delimiter=",", dtype=float)

    z = arr[0, :-1]  # Initial feasible point, excluding the last element
    c = arr[1, :-1]  # Cost vector, excluding the last element
    b = arr[2:, -1]  # Constraint vector, last column excluding the top two elements
    A = arr[2:, :-1]  # Matrix A, excluding the last column and top two rows

    z = get_any_vertex(A,b,z)
    print(z,np.matmul(c.T,z))
    z = get_opt_vertex(A,z,c,b)

if __name__ == "__main__":
    main()