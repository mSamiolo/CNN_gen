struct Perceptron {
    weight: Vec<f64>,
    bias: f64,
}

impl Perceptron {
    // Setting Weights
    
    fn new(inputs: usize, b: f64 ) -> Self {
        Self { 
            weight: vec![0.0;inputs], 
            bias: b
        }
    }

    fn set_weights(&mut self, w_init: f64) {
       self.weight.fill(w_init);
    }

    // Triggering behaviour
    fn sigmoid(&mut self, x: f64) -> f64 {
        1.0/(1.0 + (-x).exp())
    }

    fn run(&mut self, x: Vec<f64>) -> f64 {
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


fn main() {
    let mut and_gate = Perceptron::new(2, 60.0);
    and_gate.set_weights(50.0);
    let result = and_gate.run(vec![0.0, 0.0]);
    println!("Outlet {:?}:", result);
}

// ----------------------------------------------------------------------- //

// struct MultiLayerPercetron {
//     bias: f64, // The bias term. The same bias is used for all neurons.
//     eta: f64,  // Learning rate 
//     //layers: Vec<usize>
// }

// impl Default for MultiLayerPercetron {
    
//     fn default() -> Self {
//         Self {
//             bias: 0.0, 
//             eta: 0.0,
//         }
//     }

// impl MultiLayerPercetron  {
// 	fn set_layers(&self, x: [f64]) {
// 		let &mut layers = x;
// 	}
// }	
// }