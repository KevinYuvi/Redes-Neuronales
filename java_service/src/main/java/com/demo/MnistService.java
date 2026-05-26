package com.demo;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.springframework.stereotype.Service;

import jakarta.annotation.PostConstruct;
import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.InputStream;
import java.util.HashMap;
import java.util.Map;

@Service
public class MnistService {

    private MultiLayerNetwork model;

    @PostConstruct
    public void loadModel() throws Exception {
        model = ModelSerializer.restoreMultiLayerNetwork(
                new File("mnist_model_java.zip")
        );

        System.out.println("Modelo Java cargado en memoria");
    }

    public Map<String, Object> predict(InputStream imageStream) throws Exception {
        BufferedImage original = ImageIO.read(imageStream);

        BufferedImage processed = preprocess(original);

        float[] pixels = new float[28 * 28];

        for (int y = 0; y < 28; y++) {
            for (int x = 0; x < 28; x++) {
                int rgb = processed.getRGB(x, y);
                int value = rgb & 0xff;

                pixels[y * 28 + x] = value / 255.0f;
            }
        }

        INDArray input = Nd4j.create(pixels, new int[]{1, 1, 28, 28});

        long start = System.nanoTime();

        INDArray output = model.output(input);

        long end = System.nanoTime();

        int prediction = Nd4j.argMax(output, 1).getInt(0);

        double confidence = output.getDouble(0, prediction) * 100;
        double timeMs = (end - start) / 1_000_000.0;

        Map<String, Object> result = new HashMap<>();
        result.put("language", "Java");
        result.put("prediction", prediction);
        result.put("confidence", Math.round(confidence * 100.0) / 100.0);
        result.put("time_ms", Math.round(timeMs * 100.0) / 100.0);

        return result;
    }

    private BufferedImage preprocess(BufferedImage original) {
        BufferedImage gray = new BufferedImage(
                original.getWidth(),
                original.getHeight(),
                BufferedImage.TYPE_BYTE_GRAY
        );

        Graphics2D g = gray.createGraphics();
        g.drawImage(original, 0, 0, null);
        g.dispose();

        Rectangle bbox = findBoundingBox(gray);

        if (bbox == null) {
            return new BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY);
        }

        BufferedImage cropped = gray.getSubimage(
                bbox.x,
                bbox.y,
                bbox.width,
                bbox.height
        );

        int maxSize = 20;

        double scale = Math.min(
                maxSize / (double) cropped.getWidth(),
                maxSize / (double) cropped.getHeight()
        );

        int newWidth = Math.max(1, (int) Math.round(cropped.getWidth() * scale));
        int newHeight = Math.max(1, (int) Math.round(cropped.getHeight() * scale));

        BufferedImage resized = new BufferedImage(
                newWidth,
                newHeight,
                BufferedImage.TYPE_BYTE_GRAY
        );

        Graphics2D rg = resized.createGraphics();
        rg.setRenderingHint(
                RenderingHints.KEY_INTERPOLATION,
                RenderingHints.VALUE_INTERPOLATION_BILINEAR
        );
        rg.drawImage(cropped, 0, 0, newWidth, newHeight, null);
        rg.dispose();

        BufferedImage canvas = new BufferedImage(
                28,
                28,
                BufferedImage.TYPE_BYTE_GRAY
        );

        Graphics2D cg = canvas.createGraphics();
        cg.setColor(Color.BLACK);
        cg.fillRect(0, 0, 28, 28);

        int x = (28 - newWidth) / 2;
        int y = (28 - newHeight) / 2;

        cg.drawImage(resized, x, y, null);
        cg.dispose();

        return canvas;
    }

    private Rectangle findBoundingBox(BufferedImage image) {
        int minX = image.getWidth();
        int minY = image.getHeight();
        int maxX = -1;
        int maxY = -1;

        for (int y = 0; y < image.getHeight(); y++) {
            for (int x = 0; x < image.getWidth(); x++) {
                int value = image.getRGB(x, y) & 0xff;

                if (value > 30) {
                    minX = Math.min(minX, x);
                    minY = Math.min(minY, y);
                    maxX = Math.max(maxX, x);
                    maxY = Math.max(maxY, y);
                }
            }
        }

        if (maxX == -1 || maxY == -1) {
            return null;
        }

        return new Rectangle(
                minX,
                minY,
                maxX - minX + 1,
                maxY - minY + 1
        );
    }
}