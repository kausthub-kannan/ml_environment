# OneRepo for all ML engineers
Setup your Machine Learning Environment in seconds

## Scrapping
Machine Learning and Neural Networks require dataset. Best way to gett data is by web scarpping.
> ***Warning*** Before scrapping any website, please be sure that it belongs to the public domain or you have been given the permission to do so

### Image Scrapping
The `image_scrapper.py` can be used to scrap Google images. Uisng this script mutiple scarpping can be performed.  
Few python libraries have to be installed. PIP can be used for the installation.

```bash
pip install selenium requests hashlib pillow
```

After installing the required libraries, run the following command in your terminal 

```bash 
python3 image_scrapper.py
```   

> **Note**
> The path of the terminal has to be in the folder containing the python file.

Now, the terminal would ask for 3 variables: `Chrome Driver Path` , `Target Folder Path` , `Number of Images to be Scrapped per query`  
Enter the absolute path of the driver and the target folder. 

Once the single initalization variables are given, list of query parameters has to be given.
For example, I want to scrap images of cats and dogs then this is how I would do it:

```
Enter the seach query to be scarpped | Enter 'exit' to start scrapping the query list : cats images
Enter the seach query to be scarpped | Enter 'exit' to start scrapping the query list : dogs images 
Enter the seach query to be scarpped | Enter 'exit' to start scrapping the query list : exit
```
Once the list of image queries or names to be scrapped are entred the scrapping starts.   
The cat and dogs images will be stored in two different folders within the target folder. `exit` indicated the end of the list.  

> **Note**
> To have better results, entering proper queries would be required. For example, entering `cats image` would give me all type of cats but `orange cats images` would give me mostly orange cat images.
