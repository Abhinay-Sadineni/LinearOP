import pandas as pd
import numpy as np
from scipy.linalg import null_space
from scipy.optimize import root_scalar
# reading data from the csv file
pres = 1e-5
df = pd.read_csv('data.csv',header=None)
c = df.iloc[0].to_numpy()[:-1]
c = c.reshape(-1, 1)
z = df.iloc[1].to_numpy()[:-1]
z = z.flatten()

A = df.iloc[2:,0:-1].to_numpy()
b = np.transpose(df.iloc[2:,-1:].to_numpy())
b = b.reshape(-1, 1)
b  = b.flatten()
n = len(z)
m = len(b)


def find_alpha(A,u,v,i):
   if(np.isclose(A[i] @ v , 0,pres)):  return 0
   equation = lambda alpha:  A[i] @ (u + alpha * v) - b[i]
   result = root_scalar(equation, x0=0.0, method='newton', xtol=pres)
   return result.root
          

def valid_tight(A,z,u,i):
   alp = find_alpha(A,z,u,i)
   if(alp == 0) : return False
   z_prime = z + alp * u
   if(np.all(np.matmul(A,z_prime)<=b)): return True
   else : return False  


if np.all(np.matmul(A,z)<=b):
   print('yes,the point is in interior')
   # Calculate the differences between AX and b
   differences = np.dot(A, z) - b

   # Find tight rows (where the differences are very close to zero)
   tight_rows = np.isclose(differences, 0,rtol=pres)
   tight_rows = tight_rows.flatten()

   # Separate tight and untight rows
   A_tight = A[tight_rows, :]
   A_untight = A[~tight_rows, :]


   # while z is not a vertex
   while (A_tight.shape[0]<z.shape[0]):
      if len(A_tight)==0:
         # Move in a random direction until you hit one or more rows, let it be standard [1,1,....]
         u = np.ones(n)
         alp = find_alpha(A,z,u,0)
         z = z + alp*u

         differences = np.dot(A, z) - b
         tight_rows = np.isclose(differences, 0,rtol=pres)
         tight_rows = tight_rows.flatten()

         # Separate tight and untight rows
         A_tight = A[tight_rows, :]
         A_untight = A[~tight_rows, :]

      else :
         u = null_space(A_tight)[:,0]
         u = u.flatten()
         #checking parallel condition

         untight_indices = []
         for i in range(m): 
            if(not tight_rows[i]): untight_indices.append(i)

         for i in untight_indices:
            if(valid_tight(A,z,u,i)): 
               alp = find_alpha(A,z,u,i)
               break

         # while(valid_tight(A,row)): row = (row + 1) % m
         # alp = find_alpha(A,z,u,row)
         # row = (row + 1) % m
         z = z + alp*u

         differences = np.dot(A,z) - b         
         tight_rows = np.isclose(differences, 0,rtol=pres)
         tight_rows = tight_rows.flatten()

         # Separate tight and untight rows
         A_tight = A[tight_rows, :]
         A_untight = A[~tight_rows, :]

else: 
   print('The point is not in interior')
   exit(1)

# now we are at a vertex, we move to an optimal point

differences = np.dot(A,z) - b
tight_rows = np.isclose(differences,0,rtol=pres)
tight_rows = tight_rows.flatten()


coef,resid, _, _ = np.linalg.lstsq(A_tight.T, c, rcond=None)
coef = coef.flatten()

# print('Vertex : ',np.round(z,decimals=4),' Cost : ',np.dot(c.flatten(),z))
while not np.all(coef > 0):
   print('Vertex : ',np.round(z,decimals=4),' Cost : ',np.dot(c.flatten(),z))

   i = np.where(coef < 0)[0][0]
   A_inv = np.linalg.inv(A_tight)
   col =  A_inv[:,i].flatten()
   col = -1 * col
   
   untight_indices = []
   for i in range(m): 
      if(not tight_rows[i]): untight_indices.append(i)

   count = 0
   for i in untight_indices:
      if(valid_tight(A,z,col,i)): 
         alp = find_alpha(A,z,col,i)
         count += 1
         break
      
   if(count == 0): break 

   z = z + alp*col

   differences = np.dot(A,z) - b
   tight_rows = np.isclose(differences,0,rtol=pres)
   tight_rows = tight_rows.flatten()
   A_tight = A[tight_rows, :]
   
   coef,resid, _, _ = np.linalg.lstsq(A_tight.T, c, rcond=None)
   coef = coef.flatten()

if(count == 0):
   print('Given problem is unbounded')
else:   
    print('Optimal Vertex : ',np.round(z,decimals=4),' Cost : ',np.dot(c.flatten(),z))