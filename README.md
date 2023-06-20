# Latent Multi-Attribute Transformer for Face Editing in Images

Official implementation for the Master Thesis project "Latent Multi-Attribute Transformer for Face Editing in Images."

## Installation Instructions

### Dependencies
The project mainly requires:
* Python 3.10
* PyTorch 1.13
* CUDA
* Gradio (optional for the UI)
 
You can create a conda environment using the provided `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate multilat
```

Alternatively, you can install the dependencies directly from the `requirements.txt` file.

### Prepare StyleGAN2 encoder and generator

* We use the pretrained StyleGAN2 encoder and generator released from paper [Encoding in Style: a StyleGAN Encoder for Image-to-Image Translation](https://arxiv.org/pdf/2008.00951.pdf). Download and save the [official implementation](https://github.com/eladrich/pixel2style2pixel.git) to `pixel2style2pixel/` directory. Download and save the [pretrained model](https://drive.google.com/file/d/1bMTNWkh5LArlaWSc_wa8VKyq2V42T2z0/view) to `pixel2style2pixel/pretrained_models/`.

* In order to save the latent codes to the designed path, we slightly modify `pixel2style2pixel/scripts/inference.py`.

    ```
    # modify run_on_batch()
    if opts.latent_mask is None:
        result_batch = net(inputs, randomize_noise=False, resize=opts.resize_outputs, return_latents=True)
        
    # modify run()
    tic = time.time()
    result_batch, latent_batch = run_on_batch(input_cuda, net, opts) 
    latent_save_path = os.path.join(test_opts.exp_dir, 'latent_code_%05d.npy'%global_i)
    np.save(latent_save_path, latent_batch.cpu().numpy())
    toc = time.time()
    ```

### Prepare the Datasets
* Prepare the training data

    To train the latent transformer, you can download the [prepared dataset](https://drive.google.com/drive/folders/1aXVc-q2ER7A9aACSwml5Wyw5ZgrgPq52?usp=sharing) from the  to the paper [A Latent Transformer for Disentangled Face Editing in Images and Videos](https://arxiv.org/pdf/2106.11895.pdf). Place it under the directory `data/` and the [pretrained latent classifier](https://drive.google.com/file/d/1K_ShWBfTOCbxBcJfzti7vlYGmRbjXTfn/view?usp=sharing), also from the authors of the same paper, to the directory `models/`. 
    ```
    sh download.sh
    ```

    You can also prepare your own training data. To achieve that, you need to map your dataset to latent codes using the StyleGAN2 encoder. The corresponding label file is also required. You can continue to use the given pretrained latent classifier. If you want to train your own latent classifier on new labels, you can use `pretraining/latent_classifier.py`. 

* Prepare the testing data
    
    To evaluate the trained model using the same data as we did, you can download the [first 1k images from FFHQ](https://drive.google.com/drive/folders/1taHKxS66YKJNhdhiGcEdM6nnE5W9zBb1), or any subset from the same folder. Since the images are in `.png` format, it is required to use the Image2StyleGAN encoder to embed the images into the latent space. Download the png files into any path, referenced as `/path/to/ffhq/pngs/`, and use the following steps to generate the embeddings in the right directory:

```
cd pixel2style2pixel/
python scripts/inference.py \
--checkpoint_path=pretrained_models/psp_ffhq_encode.pt \
--data_path=/path/to/ffhq/pngs/ \
--exp_dir=../data/test/ \
--test_batch_size=1
```

### Original Single Transfomers
If you want to reproduce the results of the baseline paper and compare it with our results, you can download the pretrained models [here](https://drive.google.com/file/d/14uipafI5mena7LFFtvPh6r5HdzjBqFEt/view) (they are automatically downloaded from the `download.sh` script and placed under the `./logs/` folder).



## Usage Instructions
