import os
import pandas as pd
import cv2
from pathlib import Path
from PIL import Image
import torch
from torchvision import utils
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm
import argparse

from torchvision.transforms import functional as F

# For the optimizer and scheduler
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR

# Add command line arguments for training and model path
parser = argparse.ArgumentParser(description="Object Detection Training")
parser.add_argument("--train", action="store_true", help="Train the model")
parser.add_argument("--model_path", type=str, default="model.pth", help="Path to save/load the model")
args = parser.parse_args()

class VideoDataset:
    def __init__(self, base_path, annotations_path, videos_path, scene='bookstore', video='video0', transform=None):
        self.base_path = Path(base_path)
        self.annotations_path = self.base_path / annotations_path / scene / video
        self.videos_path = self.base_path / videos_path / scene / video
        self.transform = transform
        self.videos_data = {}
        self.columns = [
            'track_id', 'xmin', 'ymin', 'xmax', 'ymax', 'frame', 
            'lost', 'occluded', 'generated', 'label'
        ]
        self.load_annotations()
        self.frame_list = self.prepare_frame_list()

    def load_annotations(self):
        annotation_file = self.annotations_path / 'annotations.txt'
        if annotation_file.exists():
            data = pd.read_csv(annotation_file, sep=' ', header=None, names=self.columns)
            self.videos_data = data
        else:
            print(f"No annotation file found for video in scene")

    def prepare_frame_list(self):
        frame_list = []
        for frame_number in self.videos_data['frame'].unique():
            frame_annotations = self.videos_data[self.videos_data['frame'] == frame_number]
            frame_list.append((frame_number, frame_annotations))
        return frame_list

    
    def __len__(self):
        # Return the total number of frames
        return len(self.frame_list)

    def __getitem__(self, idx):
        # Retrieve the frame and annotations by index
        frame_number, annotations = self.frame_list[idx]
        video_path = self.videos_path / 'video.mov'
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise RuntimeError(f"Failed to grab frame {frame_number} from {video_path}")

        # Convert frame to RGB and then to a PIL Image
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)

        # Convert annotations to tensor
        boxes = torch.tensor(annotations[['xmin', 'ymin', 'xmax', 'ymax']].values, dtype=torch.float32)
        labels = torch.tensor(annotations['track_id'].values, dtype=torch.int64)  # Assuming track_id corresponds to label

        # Apply transformations to the image only if a transform is set
        if self.transform:
            frame = self.transform(frame)

        # Initialize a padded set of boxes and labels
        max_annotations = max(len(ann) for _, ann in self.frame_list)
        padded_boxes = torch.zeros((max_annotations, 4), dtype=torch.float32)
        padded_labels = torch.zeros((max_annotations,), dtype=torch.int64)

        # Copy actual annotations into the padded tensors
        num_annotations = len(annotations)
        if num_annotations > 0:
            padded_boxes[:num_annotations] = boxes
            padded_labels[:num_annotations] = labels

        # Create the target dictionary
        target = {'boxes': padded_boxes, 'labels': padded_labels}

        return frame, target  # Return a tuple with the image and the target dictionary


    
    def get_video_annotations(self, scene, video):
        return self.videos_data.get((scene, video), pd.DataFrame())

    def save_frame_with_annotations(self, scene, video, frame_number, output_path):
        '''
        To help verify the data set, i can check individual annotations through the frame number and see if it matches manually.
        '''
        video_path = self.videos_path / scene / video / 'video.mov'
        df = self.get_video_annotations(scene, video)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Error opening video file {video_path}")
            return None

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to grab frame {frame_number}")
            return None
        cap.release()

        df_frame = df[df['frame'] == frame_number]
        for index, row in df_frame.iterrows():
            cv2.rectangle(frame, (int(row['xmin']), int(row['ymin'])), (int(row['xmax']), int(row['ymax'])), (0, 255, 0), 2)
            cv2.putText(frame, row['label'], (int(row['xmin']), int(row['ymin']) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        output_file_path = Path(output_path) / f'frame_{frame_number}.jpg'
        cv2.imwrite(str(output_file_path), frame)
        return output_file_path

class ObjectDetectionModel:
    def __init__(self, num_classes):
        self.model = self.create_model(num_classes)

    def create_model(self, num_classes):
        # Load the pre-trained model with default weights
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        model = fasterrcnn_resnet50_fpn(weights=weights)

        # Get the number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        
        # Replace the pre-trained head with a new one (this also replaces the classifier)
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        return model
    
    def get_transform(self):
        # Define the data transformations for the training and validation sets
        transform = transforms.Compose([
            transforms.Resize((800, 800)), # This will work on a PIL Image
            transforms.ToTensor(),         # This will convert the PIL Image to a Tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform

class Trainer:
    def __init__(self, model, optimizer, lr_scheduler, device):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device
    
    def train_one_batch(model, images, targets, optimizer):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # Backpropagation
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        return losses.item()

    def train_one_epoch(self, data_loader, epoch):
        self.model.train()
        for i, (images, targets) in enumerate(data_loader):
            if i == 0:  # Just to print the first batch
                print(images)   # Should show a batch of images
                print(targets)  # Should show corresponding batch of targets
            images = list(image.to(self.device) for image in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # Backpropagation
            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()

        print(f"Epoch #{epoch} loss: {losses.item()}")

    def validate(self, data_loader):
        self.model.eval()
        for images, targets in data_loader:
            images = list(image.to(self.device) for image in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            with torch.no_grad():
                prediction = self.model(images)

            # Compare predictions with targets to calculate accuracy or other metrics
            # ...

    def train(self, train_loader, val_loader, num_epochs):
        for epoch in range(num_epochs):
            self.train_one_epoch(train_loader, epoch)
            self.validate(val_loader)
            self.lr_scheduler.step()

# ... [the rest of your imports and class definitions]

if __name__ == "__main__":
    base_path = 'stanford_campus_dataset'
    num_classes = 7  # 6 classes + background
    model_path = args.model_path
    object_detection_model = ObjectDetectionModel(num_classes)

    # Setup the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    object_detection_model.model.to(device)

    # Check for a saved model
    if os.path.exists(model_path) and not args.train:
        print(f"Loading model from {model_path}")
        object_detection_model.model.load_state_dict(torch.load(model_path, map_location=device))
        object_detection_model.model.eval()  # if you're loading for inference/testing
    else:
        # Create the dataset with transformations
        dataset = VideoDataset(
            base_path=base_path,
            annotations_path='annotations',
            videos_path='videos',
            scene='bookstore',
            video='video0',
            transform=object_detection_model.get_transform()
        )

        # Split dataset into train and validation sets
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        # Create the DataLoaders for our training and validation sets
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

        # Define an optimizer
        optimizer = SGD(object_detection_model.model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

        # Define a learning rate scheduler
        lr_scheduler = StepLR(optimizer, step_size=3, gamma=0.1)

        # Trainer instance
        trainer = Trainer(
            model=object_detection_model.model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            device=device
        )

        # Training loop with tqdm progress bar
        num_epochs = 10  # Define the number of epochs
        for epoch in range(num_epochs):
            with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}", unit='batch') as pbar:
                for batch in train_loader:
                    # Unpack the images and targets from the batch
                    images, targets = batch

                    # Move the images and targets to the device
                    images = [image.to(device) for image in images]
                    print(targets)
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                    loss_dict = object_detection_model.model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())

                    optimizer.zero_grad()
                    losses.backward()
                    optimizer.step()

                    pbar.set_postfix({'loss': losses.item()})
                    pbar.update(1)
            
            lr_scheduler.step()

        # Save the model after training
        torch.save(object_detection_model.model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
