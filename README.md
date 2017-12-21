# CNNArt [![Build Status](https://semaphoreci.com/api/v1/thomaskuestner/cnnart/branches/master/shields_badge.svg)](https://semaphoreci.com/thomaskuestner/cnnart) [![Waffle.io - Columns and their card count](https://badge.waffle.io/thomaskuestner/CNNArt.svg?columns=all)](https://waffle.io/thomaskuestner/CNNArt) 
Automatic and reference-free MR artifact detection

Please refer to the publications:
- [Küstner, T., Liebgott, A., Mauch, L., Martirosian, P., Bamberg, F., Nikolaou, K., Yang B., Schick F. & Gatidis, S. (2017). Automated reference-free detection of motion artifacts in magnetic resonance images. Magnetic Resonance Materials in Physics, Biology and Medicine, 1-14.](https://link.springer.com/article/10.1007/s10334-017-0650-z)

## Usage
1. define database layout in `config/database/_NAME_OF_DATABASE_.csv` (as specified in param.yml -> MRdatabase)
2. edit parameters in `config/param.yml`
3. run code via `main.py`
