mod perceptron;
use perceptron::Perceptron;

fn main() {
    let mut and_gate = Perceptron::new(2, -10.0);
    and_gate.set_weights(20.0);
    let result = and_gate.run(vec![0.0, 0.0]);
    println!("Outlet {:?}:", result.round());
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