use burn::backend::autodiff::Autodiff;
use burn::backend::cpu::Cpu;
use burn::module::Module;
use burn::tensor::{Device, Distribution, Tensor};

type MyBackend = Autodiff<Cpu>;
struct Person {
    age: f32,
    income: f32,
}
fn main() {
    let device: Device<MyBackend> = Default::default();
    let model = LinearModel::new(&device);

    let person1 = Person {
        age: 25.0,
        income: 500000.0,
    };
    let person2 = Person {
        age: 20.0,
        income: 240000.0,
    };
    let person3 = Person {
        age: 40.0,
        income: 200000.0,
    };

    let input: Tensor<MyBackend, 2> = [
        [person1.age, person1.income],
        [person2.age, person2.income],
        [person3.age, person3.income],
    ]
    .into();
    let output = model.forward(input);
    println!("Predictions from lesson 1: \n{output}\n");
}

#[derive(Module, Debug, Clone)]
struct LinearModel {
    weights: Tensor<MyBackend, 2>,
    bias: Tensor<MyBackend, 2>,
}

impl LinearModel {
    pub fn new(device: &Device<MyBackend>) -> Self {
        Self {
            weights: Tensor::random([2, 1], Distribution::Uniform(-1.0, 1.0), device),
            bias: Tensor::zeros([1, 1], device),
        }
    }
    pub fn forward(&self, input: Tensor<MyBackend, 2>) -> Tensor<MyBackend, 2> {
        return input.matmul(self.weights.clone()) + self.bias.clone();
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sum_and_product() {
        let t1: Tensor<MyBackend, 2> = [[2.0, 1.0], [2.0, 5.0]].into();
        let t2: Tensor<MyBackend, 2> = [[1.0, 5.0], [3.0, 2.0]].into();
        let sum_result = t1.clone() + t2.clone();
        let sum_values = sum_result.into_data().to_vec::<f32>().expect("testing");
        assert_eq!(sum_values, vec![3.0, 6.0, 5.0, 7.0]);

        let multiply_result = t1 * t2;
        let multi_values = multiply_result
            .into_data()
            .to_vec::<f32>()
            .expect("testing");
        assert_eq!(multi_values, vec![2.0, 5.0, 6.0, 10.0]);
    }
}
