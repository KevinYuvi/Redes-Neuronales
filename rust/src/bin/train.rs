use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{
    loss, Conv2dConfig, Linear, Module, Optimizer, VarBuilder, VarMap, SGD,
};

struct Net {
    conv1: candle_nn::Conv2d,
    conv2: candle_nn::Conv2d,
    fc1: Linear,
    fc2: Linear,
}

impl Net {
    fn new(vs: VarBuilder) -> Result<Self> {
        let cfg = Conv2dConfig {
            padding: 1,
            ..Default::default()
        };

        let conv1 = candle_nn::conv2d(1, 16, 3, cfg, vs.pp("conv1"))?;
        let conv2 = candle_nn::conv2d(16, 32, 3, cfg, vs.pp("conv2"))?;
        let fc1 = candle_nn::linear(32 * 7 * 7, 128, vs.pp("fc1"))?;
        let fc2 = candle_nn::linear(128, 10, vs.pp("fc2"))?;

        Ok(Self { conv1, conv2, fc1, fc2 })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = xs.reshape((xs.dim(0)?, 1, 28, 28))?;
        let xs = self.conv1.forward(&xs)?.relu()?.max_pool2d(2)?;
        let xs = self.conv2.forward(&xs)?.relu()?.max_pool2d(2)?;
        let xs = xs.flatten_from(1)?;
        let xs = self.fc1.forward(&xs)?.relu()?;
        let xs = self.fc2.forward(&xs)?;
        Ok(xs)
    }
}

fn accuracy(logits: &Tensor, labels: &Tensor) -> Result<f32> {
    let predictions = logits.argmax(1)?;
    let labels_u32 = labels.to_dtype(DType::U32)?;

    let correct = predictions
        .eq(&labels_u32)?
        .to_dtype(DType::F32)?
        .sum_all()?
        .to_scalar::<f32>()?;

    Ok(100.0 * correct / labels.dim(0)? as f32)
}

fn main() -> Result<()> {
    let device = Device::Cpu;

    println!("ENTRENAMIENTO RUST");
    println!("{}", "-".repeat(40));

    let dataset = candle_datasets::vision::mnist::load()?;

    let train_images = dataset.train_images.to_device(&device)?;
    let train_labels = dataset.train_labels.to_device(&device)?;
    let test_images = dataset.test_images.to_device(&device)?;
    let test_labels = dataset.test_labels.to_device(&device)?;

    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = Net::new(vs)?;

    let mut optimizer = SGD::new(varmap.all_vars(), 0.05)?;

    let batch_size = 64;
    let epochs = 1;

    let start_train = std::time::Instant::now();

    for epoch in 0..epochs {
        let train_size = train_images.dim(0)?;

        for batch_start in (0..train_size).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(train_size);
            let batch_len = batch_end - batch_start;

            let images = train_images.narrow(0, batch_start, batch_len)?;
            let labels = train_labels.narrow(0, batch_start, batch_len)?;

            let logits = model.forward(&images)?;
            let loss = loss::cross_entropy(&logits, &labels)?;

            optimizer.backward_step(&loss)?;
        }

        println!("Epoch {} completado", epoch + 1);
    }

    let train_time = start_train.elapsed().as_secs_f64();

    let start_eval = std::time::Instant::now();

    let test_logits = model.forward(&test_images)?;
    let acc = accuracy(&test_logits, &test_labels)?;

    let eval_time = start_eval.elapsed().as_secs_f64();

    varmap.save("mnist_model.safetensors")?;

    println!("Accuracy: {:.2}%", acc);
    println!("Tiempo entrenamiento: {:.2} segundos", train_time);
    println!("Tiempo inferencia/evaluacion: {:.4} segundos", eval_time);
    println!("Modelo guardado: mnist_model.safetensors");

    Ok(())
}