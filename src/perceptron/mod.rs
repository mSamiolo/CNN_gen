use rand::{self, Rng};
#[derive(Debug, Clone)]
pub struct Perceptron {
    pub weight: Vec<f64>,
    bias: f64,
}

impl Perceptron {
    // Setting Weights
    pub fn new(inputs: usize, bias: f64) -> Self {
        Self {
            weight: vec![0.0; inputs +1 ],
            bias,
        }
    }

    pub fn set_weights(&mut self, w_init: f64) {
        // Fill the weights array with random number
        for i in 0..self.weight.len() {
            let mut rng = rand::thread_rng();
            self.weight[i] = rng.gen_range(0.0..w_init);
        }
       // self.weight.fill(w_init);
    }

    // Triggering behaviour
    pub fn sigmoid(&mut self, x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    pub fn run(&mut self, x: Vec<f64>) -> f64 {
        // inizialization node_value
        let mut node_value = 0.0;
        // weighted summatory of inlets
        for i in 0..self.weight.len() -1 {
            node_value += self.weight[i] * x[i];
        }
        node_value += self.bias * self.weight.last().unwrap();
        self.sigmoid(node_value)
    }
    
    // AND GATE
    #[allow(unused)]
    pub fn run_or_gate_perceptron(x: Vec<f64>) {
        let mut and_gate = Perceptron::new(2, 0.0);
        and_gate.set_weights(10.0);
        let result = and_gate.run(x);
        println!("Outlet {:?}:", result);
}
}
