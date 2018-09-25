# -*- coding: utf-8 -*-

from scipy import linalg
import numpy as np
from random import randint
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from scipy import linalg

##################################################
#MATRIX MULTIPLICATION DEFINITION
##################################################
def matmult(a,b):
    zip_b = zip(*b)
    # uncomment next line if python 3 : 
    zip_b = list(zip_b)
    return [[sum(ele_a*ele_b for ele_a, ele_b in zip(row_a, col_b)) 
             for col_b in zip_b] for row_a in a]

##################################################
#CONTROL PANEL
##################################################
A = [[1, 1],[1, 2]]
B=[[1],[0]]
orginal_vector = 2*( B/ np.linalg.norm(B) )

origin = B
##################################################
#FOR LOOP TO SIMULATE NUMBER OF LINEAR 
# TRANSFORMATIONS
##################################################
ratio_list = []
step_axis = []
ratio_list_inverse = []

for i in range(1,300):
    if i % 9 == 0:
        print ('plotting..')
    C = matmult(A, B)
    vector_length = 1#((C[0][0])**2 + (C[1][0])**2)**(1/2)
    C_norm = [[C[0][0] / vector_length], [C[1][0] / vector_length]]
    color2 = '#{:06x}'.format(randint(0, 256**3))
    plt.quiver(*origin, C_norm[0][0], C_norm[1][0], color=color2, scale=10)
    ratio =  C[1][0] /  C[0][0]
    ratio_inverse = C[0][0] /  C[1][0]
    ratio_list.append(ratio)
    ratio_list_inverse.append(ratio_inverse)
    step_axis.append(i-1)
    B = C_norm #Change vector being transformed to prevoiously transformed vector

###################################################
# PRINTING THE DIFFERENT GRAPHS
###################################################
ori  = orginal_vector
plt.quiver(*origin, ori[1,0], ori[0,0], color='r', scale=20)
plt.title('Plotting vectors to show flow', size =20);
plt.xlabel('postion for Y', size =20)
plt.ylabel('Position for X', size = 20);
plt.show()

plt.plot(step_axis, ratio_list, 'o', color='black');
plt.title('FLOW = Ratio (Y/X)', size =20);
plt.xlabel('Number of Multiplications by A', size =20)
plt.ylabel('K, if X=K*Y', size = 20);
plt.show()


plt.plot(step_axis, ratio_list_inverse, 'o', color='black');
plt.title('FLOW = Ratio inverse (X/Y)', size=20);
plt.xlabel('Number of Multiplications by A', size = 20);
plt.ylabel('K, if K*X=Y', size = 20);
plt.show()

##################################################################################


##ANOTHER WAY OF RUNNING THE CODE:
'''
A = np.matrix([[7, 3],[3, -1]])
B = np.matrix([[3],[2]])
C=0
ratio_list = []
step_axis = []
for i in range(1,100):
    #C = np.matmul(A, B)
    C = np.matmul(A,B)
    C_norm = C / np.linalg.norm(C)
    color2 = '#{:06x}'.format(randint(0, 256**3))
    plt.quiver(*origin, C_norm[0,0], C_norm[1,0], color=color2, scale=10)
    plt.quiver(*origin, C[0,0], C[1,0], color=color2, scale=1000)
    ratio =  C[0,0] /  C[1,0]
    ratio_list.append(ratio)
    step_axis.append(i-1)
    B = C_norm #Change vector being transformed to prevoiously transformed vector





print (ratio_list)
ori  = orginal_vector
plt.quiver(*origin, ori[0,0], ori[1,0], color='r', scale=10)
plt.show()

plt.plot(step_axis, ratio_list, 'o', color='black');
plt.title('Ratio (Y/X)', size =20);
plt.show()
'''
'''
plt.plot(step_axis, ratio_list_inverse, 'o', color='black');
plt.title('Ratio inverse (X/Y)', size=20)
plt.show()
'''