#![feature(test)]

#[macro_use]
extern crate ndarray;
extern crate ndarray_rand;
extern crate rand;
extern crate test;

use ndarray::prelude::*;
use ndarray::expr::{ArrayViewExpr, BinaryOpExpr, Expression, ExpressionExt};
use ndarray_rand::RandomExt;
use rand::{Rng, SeedableRng, StdRng};
use rand::distributions::Range;
use test::Bencher;

const SEED: &[usize] = &[1, 2, 3];
const SHAPE2: (usize, usize) = (2000, 2000);

fn create_input<R: Rng>(rng: &mut R) -> Array2<f64> {
    Array2::random_using(SHAPE2, Range::new(-10., 10.), rng)
}

#[bench]
fn single_op_normal_evaluation(bencher: &mut Bencher) {
    let mut rng: StdRng = SeedableRng::from_seed(SEED);
    let a = create_input(&mut rng);
    let b = create_input(&mut rng);
    bencher.iter(|| &a + &b)
}

#[bench]
fn single_op_expr(bencher: &mut Bencher) {
    let mut rng: StdRng = SeedableRng::from_seed(SEED);
    let a = create_input(&mut rng);
    let b = create_input(&mut rng);
    bencher.iter(|| (a.as_expr() + b.as_expr()).eval())
}

#[bench]
fn two_ops_normal_evaluation(bencher: &mut Bencher) {
    let mut rng: StdRng = SeedableRng::from_seed(SEED);
    let a = create_input(&mut rng);
    let b = create_input(&mut rng);
    let c = create_input(&mut rng);
    bencher.iter(|| &a + &(&b * &c))
}

#[bench]
fn two_ops_zip(bencher: &mut Bencher) {
    let mut rng: StdRng = SeedableRng::from_seed(SEED);
    let a = create_input(&mut rng);
    let b = create_input(&mut rng);
    let c = create_input(&mut rng);
    bencher.iter(|| {
        let mut out = unsafe { Array2::uninitialized(SHAPE2) };
        azip!(mut out, a, b, c in {
            *out = a + (b * c);
        });
        out
    })
}

#[bench]
fn two_ops_expr(bencher: &mut Bencher) {
    let mut rng: StdRng = SeedableRng::from_seed(SEED);
    let a = create_input(&mut rng);
    let b = create_input(&mut rng);
    let c = create_input(&mut rng);
    bencher.iter(|| (a.as_expr() + (b.as_expr() * c.as_expr())).eval())
}

#[bench]
fn two_ops_expr_fn(bencher: &mut Bencher) {
    let mut rng: StdRng = SeedableRng::from_seed(SEED);
    let a = create_input(&mut rng);
    let b = create_input(&mut rng);
    let c = create_input(&mut rng);
    bencher.iter(|| {
        (::std::ops::Add::add(a.as_expr(), ::std::ops::Mul::mul(b.as_expr(), c.as_expr())).eval())
    })
}

#[bench]
fn two_ops_expr_manual(bencher: &mut Bencher) {
    let mut rng: StdRng = SeedableRng::from_seed(SEED);
    let a = create_input(&mut rng);
    let b = create_input(&mut rng);
    let c = create_input(&mut rng);
    bencher.iter(|| {
        BinaryOpExpr::new(
            std::ops::Add::add,
            ArrayViewExpr::new(a.view()),
            BinaryOpExpr::new(
                std::ops::Mul::mul,
                ArrayViewExpr::new(b.view()),
                ArrayViewExpr::new(c.view()),
            ).unwrap(),
        ).unwrap()
            .eval()
    })
}
