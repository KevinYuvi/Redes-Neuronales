use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{Conv2dConfig, Linear, Module, VarBuilder, VarMap};
use image::{Rgb, RgbImage};
use rand::Rng;

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

fn main() -> Result<()> {
    let device = Device::Cpu;

    println!("PREDICCION RUST");
    println!("{}", "-".repeat(40));

    let dataset = candle_datasets::vision::mnist::load()?;
    let test_images = dataset.test_images.to_device(&device)?;
    let test_labels = dataset.test_labels.to_device(&device)?;

    let mut varmap = VarMap::new();

    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = Net::new(vs)?;

    varmap.load("mnist_model.safetensors")?;

    let mut rng = rand::thread_rng();
    let index: usize = rng.gen_range(0..10000);

    let sample_image = test_images.narrow(0, index, 1)?;
    let sample_label = test_labels.narrow(0, index, 1)?;

    let logits = model.forward(&sample_image)?;
    let prediction = logits.argmax(1)?;

    let pred_vec = prediction.to_vec1::<u32>()?;
    let label_vec = sample_label.to_vec1::<u8>()?;

    let real = label_vec[0];
    let pred = pred_vec[0];

    let pixels = sample_image.flatten_all()?.to_vec1::<f32>()?;

    let scale: usize = 10;

    let width = (28 * scale) as u32;
    let height = (28 * scale) as u32;

    let mut img = RgbImage::new(width, height);

    for y in 0usize..28 {
        for x in 0usize..28 {
            let value = pixels[y * 28 + x];
            let gray = (value * 255.0).clamp(0.0, 255.0) as u8;

            for dy in 0usize..scale {
                for dx in 0usize..scale {
                    let px = (x * scale + dx) as u32;
                    let py = (y * scale + dy) as u32;

                    img.put_pixel(px, py, Rgb([gray, gray, gray]));
                }
            }
        }
    }

    let filename = format!("prediccion_rust_real_{}_pred_{}.png", real, pred);

    img.save(&filename)?;

    println!("Imagen usada del dataset MNIST: {}", index);
    println!("Real: {}", real);
    println!("Prediccion Rust: {}", pred);
    println!("Imagen guardada como: {}", filename);

    Ok(())
}