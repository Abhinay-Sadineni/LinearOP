#Assumption
#1. Polytope is non-degenerate.
#2. Polytope is bounded
#3. Rak of A is n 


#given feasible point z
import np as numpy

def get_any_vertex(A,b,c,z):

    if np.dot(A,z) <= b:
        #do nothing
       print("intial point is feasible")
    
    else:
        return None

    while( np.dot(A,z) != b):
         #A1 = tight rows
         #A2 =untight rows
         


    

    
        


