use burn::backend::Wgpu;
use burn::backend::autodiff::Autodiff;
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::optim::{GradientsParams, Optimizer, SgdConfig};
use burn::tensor::backend::Backend;
use burn::tensor::{Device, Tensor};
mod person;
use person::Person;
fn main() {
    type MyBackend = Autodiff<Wgpu>;
    let device: Device<MyBackend> = Default::default();
    let mut model = LinearModel::<MyBackend>::new(&device);
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
        "Loss BEFORE update: {:.4}",
        loss.clone().into_data().to_vec::<f32>().expect("loss")[0]
    );

    let learning_rate = 0.1_f64;
    let mut optim = SgdConfig::new().init::<MyBackend, LinearModel<MyBackend>>();
    println!("Starting training...");
    for epoch in 0..100 {
        let _output = model.forward(input.clone());
        let _loss = mse_loss(_output, targets.clone());
        let loss_value = _loss.clone().into_data().to_vec::<f32>().expect("loss")[0];

        // from_grads takes ownership of the gradient map
        let grads = GradientsParams::from_grads::<MyBackend, LinearModel<MyBackend>>(
            _loss.backward(),
            &model,
        );

        // Optimizer handles the weight update safely
        model = optim.step(learning_rate, model, grads);
        if epoch % 20 == 0 {
            println!("Epoch {}: Loss = {:.4}", epoch, loss_value);
        }
    }
    println!("Training complete!");
}
#[derive(Module, Debug)]
struct LinearModel<B: Backend> {
    layer: Linear<B>,
}
impl<B: Backend> LinearModel<B> {
    pub fn new(device: &Device<B>) -> Self {
        Self {
            layer: LinearConfig::new(2, 1).init(device),
        }
    }
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        self.layer.forward(input)
    }
}
pub fn mse_loss<B: Backend>(predictions: Tensor<B, 2>, targets: Tensor<B, 2>) -> Tensor<B, 1> {
    let diff = predictions - targets;
    let squared = diff.clone() * diff.clone();
    squared.sum()
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sum_and_product() {
        type MyBackend = Autodiff<Wgpu>;

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
