## Dependencies instruction
- Use `dicom` in version **0.9.9** in this project. The newest version of this lib is renamed as `pydicom` and some functions may not work as the same as the older version. 

## Code errors fix
### `Import` errors
- Importing the inside denpendencies. For example, in `CNNArt/utils/tfrecord/training/create_dataset.py`
```
from medio import convert_tf
```
will cause an ImportError. It may because this `tfrecord` folder was a independent project. The base path in Pycharm is `.../tfrecord`. Now after it's moved into CNNArt project,  the base path in Pycharm becomes `.../CNNArt`. So you should import it from base path, like 
```
from utils.tfrecord.medio import convert_tf
```
## Goods start for the new developers
- Convert `DICOM` files to `TFRecord` and start training: Try `CNNArt/utils/tfrecord`. 

## Fix log
- 