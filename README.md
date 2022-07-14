# BMNet+ Inference

How to use: 

- Decide on the test_bmnet+.yaml file on the amount of exemplars in exemplar_number: X.
- Put your image into test folder (assuming the image is file.jpg/file.png) and create a folder with the same name as your file.
- Crop some exemplars from the image and put the images into the folder created before.
- Run the inference py using the command `python inference.py --cfg config/test_bmnet+.yaml --img FILENAME`