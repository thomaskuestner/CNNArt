## Readme for Gadget part

### Files introduction
- `cnnart_dcm.xml` can transform `h5` after `CNNArt` to `dicom` files. Saved in `/usr/local/share/gadgetron/config` in Docker.
- `cnnart_h5.xml` can transform `h5` after `CNNArt` to `h5` files. Saved in `/usr/local/share/gadgetron/config` in Docker.
- `gadget_cnnart.py`:  XML file will call this file first. Then `gadget_cnnart.py` will use CNNArt part.In Docker, this file will be copied to `/opt/data`. Therefore it will be easier to modify it according to the various demands. 
- `cnnart_for_gadgetron.py`: The file is used to access CNNArt. It's saved in `/opt/CNNArt/Gadgetron`. The user should have no need to change it. So keep it as simple as possible and independent on the different demand. 

### Command
1. Copy Dockerfile to a empty folder.   
2. Build Docker: `sudo docker build -t cnnart_gadgetron .`  
3. Start a container: `sudo docker run -t --name gt1 --detach --volume $(pwd):/opt/data cnnart_gadgetron`  
4. Enter the container `sudo docker exec -ti gt1 bash`  
5. Copy `gadget_cnnart.py` from `CNNArt/Gadgetron`. Copy `param_minimal.yml` from `CNNArt/config`. Copy model and parameters file, too. All save in docker `/opt/data`. So you could modify it without `sudo`. 
6. Gadgetron client should run in the background. Use `top` to ensure it. If not, use `gadgetron &`. 

For example, in Docker `/opt/data` there is a dicom file `meas_MID01411_FID137567_t1_tse_tra_Kopf.dat`. Sometimes, Gadgetron will crash... If it happens, just repeat step 5. 

7. DICOM -> H5: `siemens_to_ismrmrd -f meas_MID01411_FID137567_t1_tse_tra_Kopf.dat -o t1_Kopf.h5`.
8. For dicom: `gadgetron_ismrmrd_client -f t1_Kopf.h5 -o result_t1_Kopt.dcm -c cnnart_dcm.xml` (It will generate seprate dcm files with time as name. Still working on this.)  
9. For ismrmrd: `gadgetron_ismrmrd_client -f t1_Kopf.h5 -o result_t1_Kopt.h5 -c cnnart_h5.xml`  => `result_t1_Kopt.h5`

Now, using CNNArt:
