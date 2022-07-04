pub struct Perceptron {
    weight: Vec<f64>,
    bias: f64,
}

impl Perceptron {
    // Setting Weights
    pub fn new(inputs: usize, b: f64) -> Self {
        Self {
            weight: vec![0.0; inputs],
            bias: b,
        }
    }

    pub fn set_weights(&mut self, w_init: f64) {
        self.weight.fill(w_init);
    }

    // Triggering behaviour
    pub fn sigmoid(&mut self, x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    pub fn run(&mut self, x: Vec<f64>) -> f64 {
        // inizialization node_value
        let mut node_value = 0.0;
        // weighted summatory of inlets
        for i in 0..self.weight.len() as usize {
            node_value += self.weight[i] * x[i];
        }
        node_value += self.bias;
        self.sigmoid(node_value)
    }
}
