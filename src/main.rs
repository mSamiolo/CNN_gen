// Shortening the module name
mod perceptron;
use perceptron::Perceptron;

/* This NN predicts the numberpads output, when it is shows the correct pattern the NN will output the numeber in that a human would associate to that numberpads */

fn main() {
    // Inizialization of the multilayer network and randoms weights alloation
    let mut m_l = MultiLayerPercetron::new(vec![7, 7, 7, 10], 1.0, 0.50);
    let init_weight = 0.50;
    m_l.set_weight(init_weight);
    m_l.print_weights();

    println!("\n//------------------------------- NN -------------------------------------- \n");
    println!(
        "The output should be 8 however the output has this noise: \n {:?}",
        m_l.run(Vec::from([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])).clone()    
    );

    // Starting the training procedure
    for i in 0..2000 {
        let mut mse = 0.0;
        mse += m_l.back_propagation(
            Vec::from([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]),
            Vec::from([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ); // 0 pattern
        mse += m_l.back_propagation(
            Vec::from([0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
            Vec::from([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ); // 1 pattern
        mse += m_l.back_propagation(
            Vec::from([1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]),
            Vec::from([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ); // 2 pattern
        mse += m_l.back_propagation(
            Vec::from([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0]),
            Vec::from([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ); // 3 pattern
        mse += m_l.back_propagation(
            Vec::from([0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0]),
            Vec::from([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ); // 4 pattern
        mse += m_l.back_propagation(
            Vec::from([1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]),
            Vec::from([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
        ); // 5 pattern
        mse += m_l.back_propagation(
            Vec::from([1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            Vec::from([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        ); // 6 pattern
        mse += m_l.back_propagation(
            Vec::from([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
            Vec::from([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
        ); // 7 pattern
        mse += m_l.back_propagation(
            Vec::from([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            Vec::from([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
        ); // 8 pattern
        mse += m_l.back_propagation(
            Vec::from([1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0]),
            Vec::from([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        ); // 9 pattern

        if i % 100 == 0 {
            println!("MSE = {}", mse);
        }
    }
    println!("\n------------------ After training  ---------------------- \n");
    
    let result_after_training = m_l.run(Vec::from([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]));
    for i in 0..10 {
        if result_after_training[i] > 0.5 {
            println!("The trained NN recognize: {}", i)
        }
    }
}


// -------------- Inizialize Neural Network structure and its associated modules ------------------------- //

#[derive(Debug, Clone)]
struct MultiLayerPercetron {
    bias: f64,                  // The bias term. The same bias is used for all neurons.
    eta: f64,                   // Learning rate
    layers: Vec<usize>,         // Array with the number of elements per layer
    values: Vec<Vec<f64>>,
    d: Vec<Vec<f64>>,
    network: Vec<Vec<Perceptron>>
}

impl MultiLayerPercetron {

    // ------------------------ Construct NN ------------------------ //

    fn new(layers: Vec<usize>, bias: f64, eta: f64) -> Self {
        // Initialization vector parameters
        let mut values = Vec::new();
        //  Initialization matrix of error terms (d = lowercase deltas) and neurons 
        let mut d = Vec::new();         
        let mut network = Vec::new();
        // Let space for the input layer
        network.push(Vec::new());

        // Start instancianting the entinties you need to store data
        for i in 0..layers.len() {
            values.push(vec![0.0; layers[i]]);
            d.push(vec![0.0; layers[i]]);
            // instaciate the network of neurons
            if i > 0 {
                // Neuron start from the second layers, and the accept the as input the number of node predent to them plus the bias input 
                network.push(vec![Perceptron::new(layers[i - 1] , bias); layers[i]]);
            }
        }

        // Initialization Multipercetron parameters
        Self {
            bias,
            eta,
            layers,
            values,
            d,
            network,
        }
    }

    // ------------------------ Weight settings ------------------------ //

    fn set_weight(&mut self, w_init: f64) {
        // Start from 1 because the first layer does not contains neurons
        for i in 1..self.layers.len() {
            for j in 0..self.layers[i]  {
                self.network[i][j].set_weights(w_init);
            }
        }
    }

    #[allow(unused)]
    fn print_weights(&self) {
        for i in 1..self.layers.len() {
            println!("Layer {}", i);
            // The second for cycle start from 1.0 because it is the second layer that contains neurons
            for j in 0..self.layers[i] {
                println!("Neuron {}, Weight {:?}", j + 1, self.network[i][j].weight)
            }
        }
    }

    // --------------------- Running algortims -------------------- //

    fn run(&mut self, x: Vec<f64>) -> &Vec<f64> {
        // Setting the first layer as the input layer
        self.values[0] = x;
        // hence from the second layer
        for i in 1..self.layers.len() {
            for j in 0..self.layers[i] {
                self.values[i][j] = self.network[i][j].run(self.values[i-1].clone());
            }
        }
        match self.values.last() {
            Some(t) => t,
            None => panic!("Problem in running the MultiLayerPercetron")
        }
    }

    // --------------------- Training algortims / Back propagation -------------------- //
    fn back_propagation(&mut self, x: Vec<f64>, y: Vec<f64>) -> f64 {

        // Feed a sample to the NN and take the output vector for a comparison with y (the real result)
        let output = self.run(x).clone();
        if output.len() != y.len() {
            panic!("The output vector is not the same length as the input vector")
        }
        
        // Inizialization of the output vector
        let mut error: Vec<f64> = Vec::new();
        let mut mse: f64 = 0.0;

        // Calculate the error term of each perceptron on each layer          
            for i in (0..self.layers.len()).rev() {
            //  penultimo ... 2 1 0
            for j in 0..self.network[i].len() {
                if i == self.layers.len() - 1 {
                    // Calculate the mse - Mean Squared Error and the delta vector (the output error terms) in the last layer
                    error.push(y[j] - output[j]);
                    self.d[i][j] = output[j] * (1.0 - output[j]) * (y[j] - output[j]);
                } else {

                    // Instanciate the forward error
                    let mut fwd_error = 0.0;
                    for k in 0..self.layers[i + 1] {
                        fwd_error += self.network[i + 1][k].weight[j] * self.d[i + 1][k];
                    }
                    self.d[i][j] = self.values[i][j] * (1.0 - self.values[i][j]) * fwd_error;
                }
            }

            // Evaluate the MSE
            match self.layers.last() {
                Some(n) => mse = error.iter().sum::<f64>().powf(2.0) / n.clone() as f64,
                None => panic!("Invalid layer value at the end of the NN"),
            }
        }

        // Calculate the deltas and update the weights
        for i in 1..self.layers.len() {
            for j in 0..self.layers[i] {
                for k in 0..self.layers[i-1] + 1 { 
                    let delta: f64;
                    // Quantify the weight correction associated to the bias term
                    if k == self.layers[i-1] {
                        delta = self.eta * self.d[i][j] * self.bias;  // --> IT is not working
                    }
                    // Quantify the weight correction  associated to the neurons
                    else {
                        delta = self.eta * self.d[i][j] * self.values[i - 1][k];
                    }
                    // Updating NN weights
                    self.network[i][j].weight[k] += delta;
                }
            }
        }
        mse
    }
}


