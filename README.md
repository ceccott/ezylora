# EzyLoRA
Simplifying init, format, BLIP caption and settings for kohya_ss LoRA training

# Install 
This script is based on [kohya_ss](https://github.com/bmaltais/kohya_ss). Follow the installation instructions for that.
Install the requirements for ezylora with `python venv` by running
```
chmod +x install.sh ezylora.py
./install.sh
```

# Run
Before running ezylora, start a session of kohya_ss from its installation folder by running:
```
./gui.sh --listen=0.0.0.0 --headless
```
Make sure kohya_ss is running by navigating to its UI at `localhost:7860`. 
From ezylora folder, call the script with the following required arguments after sourcing the environment:
```
source venv/bin/activate
./ezylora.py --lora_name 'your LoRA name' --src_path 'path/to/training/image/set'
```
For a full list of arguments, call `ezylora -h`:

```
usage: ezylora.py [-h] --lora_name LORA_NAME --src_path SRC_PATH [--dst_path DST_PATH] [--rename_pics] [--endpoint ENDPOINT]

Init, format, BLIP caption and training settings, for kohya_ss LoRA training

options:
  -h, --help            show this help message and exit
  --lora_name LORA_NAME
                        name for LoRa model
  --src_path SRC_PATH   training images folder, if supplied num_pics is automatically set
  --dst_path DST_PATH   destination path for the generated folder tree and files
  --rename_pics         rename pictures according to <LoRA name>_<number> pattern if supplied
  --endpoint ENDPOINT   kohya_ss endpoint
  ```

# Output
The output of the Run step is a folder tree containing all the source images, together with their BLIP caption txt pairs and a `LoraSettings.json` file to be loaded in kohya_ss to start the training.
```
my_lora
├── image
│   └── 100_my_lora
│       └── my_lora_0.jpg
│           my_lora_0.txt
│           ...
├── log
├── LoraSettings.json
└── model
```

# Kohya_ss training
To train the LoRA, navigate to the kohya_ss web UI at `localhost:7860` and load the `LoraSettings.json` configuration file by setting its absolute path under LoRA->Training->Configuration File and clicking Load 💾.

![image](https://github.com/ceccott/ezylora/assets/5775579/32d8ed1c-7bbd-484c-9a01-7e205edb5bd6)
