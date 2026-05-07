use burn::backend::autodiff::Autodiff;
use burn::backend::cpu::Cpu;
use burn::nn::{Linear, LinearConfig};
use burn::module::Module;
use burn::optim::{GradientsParams, Optimizer, SgdConfig};
use burn::tensor::{Device, Tensor};

type MyBackend = Autodiff<Cpu>;
struct Person {
    age: f32,
    income: f32,
}
fn main() {
    let device: Device<MyBackend> = Default::default();
    let mut model = LinearModel::new(&device);

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
    let max_age = 40.0;
    let max_income = 500000.0;
    let input: Tensor<MyBackend, 2> = [
        [person1.age / max_age, person1.income / max_income],
        [person2.age / max_age, person2.income / max_income],
        [person3.age / max_age, person3.income / max_income],
    ]
    .into();
    let output = model.forward(input.clone());
    let targets: Tensor<MyBackend, 2> = [[1.0], [0.0], [0.0]].into();
    let loss = mse_loss(output, targets.clone());
    println!(
        "Loss BEFORE update: {}",
        loss.clone().into_data().to_vec::<f32>().expect("loss")[0]
    );
    let learning_rate = 0.1_f64;
    let mut optim = SgdConfig::new().init::<MyBackend, LinearModel>();

    println!("Starting training...");
    for epoch in 0..100 {
        let output = model.forward(input.clone());
        let loss = mse_loss(output, targets.clone());
        
        let loss_value = loss.clone().into_data().to_vec::<f32>().expect("loss")[0];
        let mut gradients = loss.backward();
        
        let grads = GradientsParams::from_module::<MyBackend, LinearModel>(&mut gradients, &model);
        model = optim.step(learning_rate, model, grads);

        if epoch % 20 == 0 {
            println!("Epoch {}: Loss = {:.4}", epoch, loss_value);
        }
    }
    println!("Training complete!");
}

#[derive(Module, Debug, Clone)]
struct LinearModel {
    layer: Linear<MyBackend>,
}

impl LinearModel {
    pub fn new(device: &Device<MyBackend>) -> Self {
        Self {
            layer: LinearConfig::new(2, 1).init(device),
        }
    }
    pub fn forward(&self, input: Tensor<MyBackend, 2>) -> Tensor<MyBackend, 2> {
        self.layer.forward(input)
    }
}

pub fn mse_loss(
    predictions: Tensor<MyBackend, 2>,
    targets: Tensor<MyBackend, 2>,
) -> Tensor<MyBackend, 1> {
    let diff = predictions - targets;
    let squared = diff.clone() * diff.clone();
    return squared.mean();
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_mse_loss() {
        let predictions: Tensor<MyBackend, 2> = [[1.0], [0.0], [0.5]].into();
        let targets: Tensor<MyBackend, 2> = [[1.0], [1.0], [0.0]].into();

        let loss = mse_loss(predictions, targets.clone());
        let loss_value = loss.into_data().to_vec::<f32>().expect("loss tensor");

        // MSE = average of (prediction - target) ** 2
        assert!((loss_value[0] - 0.4167_f32).abs() < 1e-4);
    }
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
