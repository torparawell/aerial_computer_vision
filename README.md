# aerial_computer_vision
Using the Stanford drone project data set, I decided to create a computer vision project using PyTorch to encapsulate a the advantages of having an aerial camera on an environment for object detection. Specifically includes the implementation of an object detection model using Faster R-CNN with a ResNet50 backbone in PyTorch. The model is designed to work on video data and can be trained on custom datasets.

## Dependencies

- Python 3.8 or higher
- PyTorch
- torchvision
- OpenCV
- pandas
- PIL
- tqdm

## Installation

Clone the repository to your local machine:

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

Install the required libraries:

```bash
pip install torch torchvision opencv-python pandas pillow tqdm
```

## Dataset

The code is configured to work with a dataset that has the following structure:

```
stanford_campus_dataset/
    annotations/
        bookstore/
            video0/
                annotations.txt
    videos/
        bookstore/
            video0/
                video.mov
```

## Usage

To train the model with the default settings, run:

```bash
python your_script.py --train
```

To use a pre-trained model and perform inference, ensure you have the `model.pth` file in your project directory and run:

```bash
python your_script.py
```

## Customization

You can customize the training by specifying the path to save or load the model using the `--model_path` argument.

## Contributing

Contributions to this project are welcome. Please open an issue first to discuss what you would like to change or add.

## License

Include a license here if applicable.

```

Replace `your-username/your-repo-name` with your actual GitHub username and repository name, and `your_script.py` with the name of your script file if it's different.

The above README provides a basic outline. You can expand on each section with more details if necessary, such as adding a section on how to prepare the dataset, more detailed usage examples, contribution guidelines, etc.
