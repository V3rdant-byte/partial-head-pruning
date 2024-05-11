# Partial Head Quantization
## Preface
This is a repo for an undergraduate thesis by Jacob Yang at University 
of British Columbia supervised by Professor Prashant Nair and a PhD student 
Muhammad Abdullah Adnan. The thesis in pdf is in the root folder [./CPEN499 Jacob's thesis.pdf](./CPEN499%20Jacob's%20thesis.pdf).
## How to set up environment
I encountered some config problem when setting up smoothquant environment.
Here is how I solve those problems.

[CUDA driver 12.3](https://www.nvidia.com/Download/index.aspx)

[CUDA Toolkit 11.6](https://developer.nvidia.com/cuda-11-6-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local)
Tried 12.3, 11.3, 11.8
cuda toolkit cannot be easily downgraded. just reinstall the system

[Anaconda](https://www.how2shout.com/how-to/install-anaconda-wsl-windows-10-ubuntu-linux-app.html#:~:text=Open%20the%20browser%20of%20your%20Windows%2010%20or,paste%20the%20link%2C%20for%20example%3A%20wget%20paste-link%20Example%3A)

Clone [Smoothquant](https://github.com/mit-han-lab/smoothquant.git)

Install smoothquant following the [readme](./smoothquant/README.md)

Use pytorch 1.2.1 and cuda 11.6
Install [torch-int](https://github.com/Guangxuan-Xiao/torch-int) following the readme

Install [cutlass](https://github.com/NVIDIA/cutlass/tree/c975e2ccbb2dbf13024568b37ffa3498ed0b3aed) check out to feature/2.10

Some issue [fixes](https://github.com/FurryMushroom/Quantization_work_of_DesignOrder/blob/main/Experiments%20on%20Smoothquant.md)

Convert the demo jupyter notebook smoothquant_opt_real_int8_demo.ipynb into python script and then run it in the virtual environment with smoothquant and torch-int installed.
## How to generate act scales
According to [Issue 60](https://github.com/mit-han-lab/smoothquant/issues/60) in the smoothquant repo, generate activation scales with lambda dataset
“ dataset = load_dataset(‘lambda’, split = ‘validation[:1000]’) ”
in smoothquant/calibration.py get_act_scales

Disable the dataset_path for the generate_act_scales.py and run the default setting

Change the path to the newly generated activation scales in test_quant.py

## Usage
To run the test, run 
```bash
python ./examples/test_quant.py
```
There are an argument to modify in the examples/test_quant.py. For example,
how many less significant heads to quantize.

Also, you need to add the head matrices to one of the outputs of the attention
layers for the python code to run.
Add the line 
```bash
attn_weights_reshaped = attn_output
```
In between the line
```bash
attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
```
and
```bash
attn_output = attn_output.transpose(1, 2)
```