use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{Conv2dConfig, Linear, Module, VarBuilder, VarMap};
use std::env;

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

        Ok(Self {
            conv1,
            conv2,
            fc1,
            fc2,
        })
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

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        println!("{{\"error\":\"Falta ruta de imagen\"}}");
        return Ok(());
    }

    let image_path = &args[1];

    let device = Device::Cpu;

    let mut varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = Net::new(vs)?;

    varmap.load("mnist_model.safetensors")?;

    let img = image::open(image_path)?
        .resize_exact(28, 28, image::imageops::FilterType::Triangle)
        .to_luma8();

    let pixels: Vec<f32> = img
        .pixels()
        .map(|p| p[0] as f32 / 255.0)
        .collect();

    let input = Tensor::from_vec(
        pixels,
        (1, 28 * 28),
        &device
    )?;

let logits = model.forward(&input)?;

// Convertir logits a probabilidades
let probs = candle_nn::ops::softmax(&logits, 1)?;

// Predicción: índice con mayor probabilidad
let prediction = probs.argmax(1)?;
let pred_vec = prediction.to_vec1::<u32>()?;

// Obtener confianza
let probs_vec = probs.flatten_all()?.to_vec1::<f32>()?;
let confidence = probs_vec[pred_vec[0] as usize] * 100.0;

println!(
    "{{\"prediction\":{},\"confidence\":{:.2}}}",
    pred_vec[0],
    confidence
);

    Ok(())
}