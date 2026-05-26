package com.demo;

import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.util.Map;

@RestController
@CrossOrigin(origins = "*")
public class PredictionController {

    private final MnistService mnistService;

    public PredictionController(MnistService mnistService) {
        this.mnistService = mnistService;
    }

    @PostMapping("/predict")
    public Map<String, Object> predict(@RequestParam("file") MultipartFile file) throws Exception {
        return mnistService.predict(file.getInputStream());
    }
}