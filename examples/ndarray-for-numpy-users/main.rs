#[macro_use]
extern crate ndarray;

use ndarray::prelude::*;

fn main() {
    
    //Creating an array of 3s
    let mut a = Array::from_elem((5,2), 3.0);
    //Here's just a simple example showing just one way to change a slice of our data.
    //I'm just assigning a new value to each set of our slice, but we could also apply a
    //scalar function to all of the values of the slice as well.
    a.slice_mut(s![..,0]).mapv_inplace(|_x| 1.0);
    //We can see here that the values did change.
    println!("\nA\n{:?}", a);

    //If we want to work on a set of data that meets a logical set of conditions we have a couple of options available.
    //We could loop through the data by using a mutable iterator in conjunction with a filter.
    for i in a.iter_mut().filter(|x| **x < 3.0){
        *i = 4.0;
    }
    //If we want to avoid the for loop we could also do the following:
    a.mapv_inplace(|x| {if x < 3.0 {4.0}else{x}});

    //We can now print our data and see that the values were modified based upon our logical condition.
    println!("\nA\n{:?}", a);

    //Let's do a rotation example next using Bunge angles and then a simple passive rotation of our
    //coordinate system. The difference between a passive and active rotation can pretty much come
    //down to whether we want to rotate our coordinate system or simply the body itself. If we
    //are rotating the body then it's an active rotation. If we are rotating the coordinate
    //system it's a passive rotation. Also, the active and passive rotation matrices differ by a simple
    //transpose operation on the rotation matrix. 

    //We're going to be going row by row here, so it makes sense to keep the standard
    //row memory stride setup
    let bunge = Array2::<f64>::from_elem((3, 4), 1.0);

    let s1:Array1<f64> = bunge.slice(s![0, ..]).iter().map(|&rho1| f64::sin(rho1)).collect();
    let c1:Array1<f64> = bunge.slice(s![0, ..]).iter().map(|&rho1| f64::cos(rho1)).collect();

    let s2:Array1<f64> = bunge.slice(s![1, ..]).iter().map(|&phi1| f64::sin(phi1)).collect();
    let c2:Array1<f64> = bunge.slice(s![1, ..]).iter().map(|&phi1| f64::cos(phi1)).collect();

    let s3:Array1<f64> = bunge.slice(s![2, ..]).iter().map(|&rho2| f64::sin(rho2)).collect();
    let c3:Array1<f64> = bunge.slice(s![2, ..]).iter().map(|&rho2| f64::cos(rho2)).collect();

    let nelems = bunge.len_of(Axis(1));
    //We're going to make this a column memory stride setup, since we'll be using the
    //first two dimensions the most often.
    let mut rmat = Array3::<f64>::zeros((3, 3, nelems).set_f(true));

    //We could also do this using iterators like seen above. However, we would be taking
    //a hit due to the fact that we aren't striding over memory instead of operating on
    //consecutive memory.
    //
    //Also, if we'd wanted to we could have also have just calculated the necessary sines and
    //cosines in this loop instead of doing it all at once like seen above.
    //However if we'd gone that route, then we'd would want to change the bunge array so that it was
    //using column strides for its memory layout.
    for i in 0..nelems{

        rmat[(0, 0, i)] = c1[i] * c3[i] - s1[i] * s3[i] * c2[i];
        rmat[(0, 1, i)] = -c1[i] * s3[i] - s1[i] * c2[i] * c3[i];
        rmat[(0, 2, i)] = s1[i] * s2[i];

        rmat[(1, 0, i)] = s1[i] * c3[i] + c1[i] * c2[i] * s3[i];
        rmat[(1, 1, i)] = -s1[i] * s3[i] + c1[i] * c2[i] * c3[i];
        rmat[(1, 2, i)] = -c1[i] * s2[i];

        rmat[(2, 0, i)] = s2[i] * s3[i];
        rmat[(2, 1, i)] = s2[i] * c3[i];
        rmat[(2, 2, i)] = c2[i];

    }

    //Here we're just seeing what our rmat initially looks like.
    //We'll have another example down below that should be identical to this value.
    println!("\nrmat\n{:?}", rmat.slice(s![.., .., 0]));

    let eye2d = Array2::<f64>::eye(3);

    let mut mat_rot = Array3::<f64>::zeros((3, 3, nelems).set_f(true));

    let mut crd_sys_rot = Array3::<f64>::zeros((3, 3, nelems).set_f(true));

    //The below shows two examples:
    //The first example mat_rot shows how to apply a rotation to a tensor
    //In our example this will just return the identity matrix because R*I*R^T = R*R^T = I
    //The second example crd_sys_rot shows how to apply a rotatation to a coordinate
    //system or even a series of vectors.
    for i in 0..nelems{
        mat_rot.slice_mut(s![.., .., i]).assign({
            &rmat.slice(s![.., .., i]).dot({
                &eye2d.dot(&rmat.slice(s![.., .., i]).t())
            })
        });
        //Since we are just multiplying by identity here our
        //coordinate system is just equal to our rotation matrix
        crd_sys_rot.slice_mut(s![.., .., i]).assign({
            &rmat.slice(s![.., .., i]).dot(&eye2d)
        });

    }

    //Here we're going to just print off the first tensor/matrix and coordinate system
    //corresponding to our n_element 3D array of matrices and coordinate systems.
    //The rest of the tensors and coordinate systems will be exactly the same. 
    println!("\ncrd_sys_rot\n{:?}", crd_sys_rot.slice(s![.., .., 0]));
    println!("\nmat_rot\n{:?}", mat_rot.slice(s![.., .., 0]));

}
