use burn::backend::autodiff::Autodiff;
use burn::backend::cpu::Cpu;
use burn::tensor::Tensor;

type MyBackend = Autodiff<Cpu>;

fn main() {
    // Create a 2D tensor from a nested slice
    let t1: Tensor<MyBackend, 2> = [[1.0, 2.0], [3.0, 4.0]].into();
    println!("t1:\n{t1}\n");

    // Create another tensor
    let t2: Tensor<MyBackend, 2> = [[5.0, 6.0], [7.0, 8.0]].into();
    println!("t2:\n{t2}\n");

    // Element-wise addition
    let sum = t1.clone() + t2.clone();
    println!("t1 + t2:\n{sum}\n");

    // Element-wise multiplication
    let product = t1.clone() * t2.clone();
    println!("t1 * t2:\n{product}\n");

    // Matrix multiplication
    let matmul = t1.matmul(t2);
    println!("t1 @ t2:\n{matmul}\n");
}
