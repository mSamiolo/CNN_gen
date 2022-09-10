# Neural Network for classification problems
Libraries to generate and manage a multilple layers Convolutional Neural Network.
Classification problem can be solved such as recognition of image or pattern or even topology optimzation. The number of layers and neurons are let to the user during the creation of the network.

## Installation
Use your Cargo package manager adding to its config file the latest verison available:
```toml
[dependencies]

cnnks = "xxx"
```

## Code snippet
You can generate and train the CNN via:
```rust
    let mut m_l = cnnks::MultiLayerPercetron::new(7, 10, vec![10,10], 1.0, 0.50);
    let init_weight = 0.50;
    m_l.set_weight(init_weight);

    println!("\n//------------------------------- NN -------------------------------------- \n");
    println!(
        "The output should be 8 however the output has this noise: \n {:?} \n",
        m_l.run(Vec::from([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))
            .clone()
    );

    // Starting the training procedure
    for i in 0..2000 {
        let mut mse = 0.0;
        mse += m_l.back_propagation(
            Vec::from([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]),
            Vec::from([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))?; // 0 pattern
        mse += m_l.back_propagation(
            Vec::from([0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
            Vec::from([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))?; // 1 pattern
        mse += m_l.back_propagation(
            Vec::from([1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]),
            Vec::from([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))?; // 2 pattern
        mse += m_l.back_propagation(
            Vec::from([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0]),
            Vec::from([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))?; // 3 pattern
        mse += m_l.back_propagation(
            Vec::from([0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0]),
            Vec::from([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]))?; // 4 pattern
        mse += m_l.back_propagation(
            Vec::from([1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]),
            Vec::from([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]))?; // 5 pattern
        mse += m_l.back_propagation(
            Vec::from([1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            Vec::from([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]))?; // 6 pattern
        mse += m_l.back_propagation(
            Vec::from([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
            Vec::from([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]))?; // 7 pattern
        mse += m_l.back_propagation(
            Vec::from([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            Vec::from([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]))?; // 8 pattern
        mse += m_l.back_propagation(
            Vec::from([1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0]),
            Vec::from([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]))?; // 9 pattern

        if i % 100 == 0 {
            println!("MSE = {}", mse);
        }
    }

    // Run the NN after the training procedure
    println!("\n------------------ After training  ---------------------- \n");
    let result_after_training = m_l.run(Vec::from([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]));
    println!("{:?}\n ", result_after_training);
```

## Test
The test unit give as input a numberpads which mean a number from 0 to 9.
During the training procedure several numberpads segments pattern are exposed, telling the corrisponding given number.
The NN are the beginning will garbage since the weight are set by the user but after the training it should return the number the numberpad meant.
It is possible to execute the test via:
```shell-session
cargo test
```