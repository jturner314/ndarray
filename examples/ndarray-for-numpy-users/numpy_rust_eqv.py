
# -*- coding: utf-8 -*-

import numpy as np

a = 3.0 * np.ones((5, 2))
a[:, 0] = 1.0
print(a)

a[a < 3.0] = 4.0
print(a)

'''
Let's do a rotation example next using Bunge angles and then a simple passive rotation of our
coordinate system. The difference between a passive and active rotation can pretty much come
down to whether we want to rotate our coordinate system or simply the body itself. If we
are rotating the body then it's an active rotation. If we are rotating the coordinate
system it's a passive rotation. Also, the active and passive rotation matrices differ by a simple
transpose operation on the rotation matrix. 

We're going to be going row by row here, so it makes sense to keep the standard
row memory stride setup
'''
bunge = np.ones((3, 4))

s1 = np.sin(bunge[0, :])
c1 = np.cos(bunge[0, :])

s2 = np.sin(bunge[1, :])
c2 = np.cos(bunge[1, :])

s3 = np.sin(bunge[2, :])
c3 = np.cos(bunge[2, :])

nelems = bunge.shape[1]

#We're going to make this a column memory stride setup since we'll be using the
#first two dimensions the most often.

rmat = np.zeros((3, 3, nelems), order='F')

'''
We could also do this using iterators like seen above. However, we would be taking
a hit due to the fact that we aren't striding over memory instead of operating on
consecutive memory.

Also, if we'd wanted to we could have also have just calculated the necessary sines and
cosines in this loop instead of doing it all at once like we did above.
However if we'd gone that route, then we'd would want to change the bunge array so that it was
using column strides for its memory layout.
'''
for i in range(nelems):
    
    rmat[0, 0, i] = c1[i] * c3[i] - s1[i] * s3[i] * c2[i]
    rmat[0, 1, i] = -c1[i] * s3[i] - s1[i] * c2[i] * c3[i]
    rmat[0, 2, i] = s1[i] * s2[i]

    rmat[1, 0, i] = s1[i] * c3[i] + c1[i] * c2[i] * s3[i]
    rmat[1, 1, i] = -s1[i] * s3[i] + c1[i] * c2[i] * c3[i]
    rmat[1, 2, i] = -c1[i] * s2[i]

    rmat[2, 0, i] = s2[i] * s3[i]
    rmat[2, 1, i] = s2[i] * c3[i]
    rmat[2, 2, i] = c2[i]

print(rmat[:, :, 0])

eye2d = np.eye(3)

mat_rot = np.zeros((3, 3, nelems), order='F')
crd_sys_rot = np.zeros((3, 3, nelems), order='F')

for i in range(nelems):
    mat_rot[:,:,i] = rmat[:,:,i].dot(eye2d.dot(rmat[:,:,i]).T)
    #Since we are just multiplying by identity here our
    #coordinate system is just equal to our rotation matrix
    crd_sys_rot[:,:,i] = rmat[:,:,i].dot(eye2d)
    
    
print(crd_sys_rot[:,:,0])
print(mat_rot[:,:,0])