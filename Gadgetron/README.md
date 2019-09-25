## Readme for Gadget part

### Files introduction
<!-- - `cnnart_dcm.xml` can transform `h5` after `CNNArt` to `dicom` files. Saved in `/usr/local/share/gadgetron/config` in Docker. -->
- `cnnart_h5.xml` can transform `h5` after `CNNArt` to `h5` files. Saved in `/usr/local/share/gadgetron/config` in Docker.
- `gadget_cnnart.py`:  XML file will call this file first. Then `gadget_cnnart.py` will use CNNArt part.In Docker, this file will be copied to `/opt/data`. Therefore it will be easier to modify it according to the various demands. 
- `cnnart_for_gadgetron.py`: The file is used to access CNNArt. It's saved in `/opt/CNNArt/Gadgetron`. The user should have no need to change it. So keep it as simple as possible and independent on the different demand. 
- `param_minimal_gadgetron.yml`: The related parameter document. 
### Usage
1. Copy Dockerfile to a empty folder.   
2. Build Docker: `sudo docker build -t cnnart_gadgetron .`  
3. Start a container: `sudo docker run -t --name gt1 --detach --volume $(pwd):/opt/data cnnart_gadgetron`. This will link current path to `\opt\data   ` in docker.  
4. Enter the container `sudo docker exec -ti gt1 bash`  
5. Change `param_minimal_gadgetron.yml` in `CNNArt/config`. Copy model and parameters file to current path, too. So they are operable in docker `/opt/data`. So you may need `sudo` permission to  modify them. 
6. Gadgetron client should run in the background. Use command `top` to ensure it. If not, an "connection to Socket error" will appear. Use `gadgetron &` to start gadgetron. 
7. If Gadgetron client is running background at start, the log file is in `\tmp\gadgetron`. If not, the log output will show in the terminal. 

For example, in Docker `/opt/data` there is a dicom file `MS_3DTSE_51005_198218855_198218860_432_20190408-172944.h5`. Sometimes, Gadgetron will crash... If it happens, just repeat step 6. 

8. For hdf5: `gadgetron_ismrmrd_client -f MS_3DTSE_51005_198218855_198218860_432_20190408-172944.h5 -o result.h5 -c cnnart_h5.xml`.  

### Output example
```
==== Gadget start ====
==== In fast_call, config loaded ====
==== In fast_call, Data() set ====
==== In fast_call, rigidPatching3D finished ====
==== Import Networks ====
==== Artifact detection ====
==== FCN, model loaded ====
==== FCN, model.compile finished ====
==== FCN, weights loaded ====
90/90 [==============================] - 1554s 17s/step
==== Result plotting ====
==== All finished ====

...
09-22 11:18:23.390 INFO [GadgetStreamController.cpp:190] Stream is closed

```