### Custom and Pretrained CNN for Identifying Car Makes

By Sem Verkerk and Tyler Fabela 



We developed and trained a custom CNN and a pretrained CNN in order to identify the car make when provided with a decent image of the exterior of a car and the make as the first token in the file name. We used a dataset from Kaggle called "60,000+ Images of Cars", downloaded it as a zip, used a CLIP filter to filter out ay faulty images to get a total of 37 makes (classes) and 9,250 images, used the data to train a custom CNN, and used the data to compare against a pretrained CNN using MobileNetV2 (warning: both CNN models are overfitted). The specifics for how the program interprets the information and how to set up the environment are fairly self-explanatory and are located within the Jupyter Notebook File within this repository, which also contains the code to run the program. Information about the dataset is below.



Location: Kaggle

Dataset Name: "60,000+ Images of Cars"

Credit to: Paul (Owner of Dataset)

Website: https://www.kaggle.com/datasets/prondeau/the-car-connection-picture-dataset/data
