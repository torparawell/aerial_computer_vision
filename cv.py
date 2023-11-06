import os
import pandas as pd
import cv2
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# If you use any other specific functions or classes from torchvision, import them here
# For example, if you use specific transforms:
from torchvision.transforms import functional as F

# For the optimizer and scheduler
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR

class VideoDataset:
    def __init__(self, base_path, annotations_path, videos_path, transform=None):
        # Initialize dataset, you might want to add additional arguments to pass transforms etc.
        self.base_path = Path(base_path)
        self.annotations_path = self.base_path / annotations_path
        self.videos_path = self.base_path / videos_path
        self.transform = transform
        self.videos_data = {}
        self.columns = [
            'track_id', 'xmin', 'ymin', 'xmax', 'ymax', 'frame', 
            'lost', 'occluded', 'generated', 'label'
        ]
        self.load_annotations()
        self.frame_list = self.prepare_frame_list()

    def load_annotations(self):
        for scene_dir in self.annotations_path.iterdir():
            if scene_dir.is_dir():
                for video_dir in scene_dir.iterdir():
                    if video_dir.is_dir():
                        annotation_file = video_dir / 'annotations.txt'
                        if annotation_file.exists():
                            data = pd.read_csv(annotation_file, sep=' ', header=None, names=self.columns)
                            self.videos_data[(scene_dir.name, video_dir.name)] = data
                        else:
                            print(f"No annotation file found for video {video_dir.name} in scene {scene_dir.name}")

    def prepare_frame_list(self):
        # Prepare a list of all frames and their corresponding annotations
        frame_list = []
        for (scene, video), df in self.videos_data.items():
            for frame_number in df['frame'].unique():
                frame_annotations = df[df['frame'] == frame_number]
                frame_list.append((scene, video, frame_number, frame_annotations))
        return frame_list
    
    def __len__(self):
        # Return the total number of frames
        return len(self.frame_list)

    def __getitem__(self, idx):
        # Retrieve the frame and annotations by index
        scene, video, frame_number, annotations = self.frame_list[idx]
        video_path = self.videos_path / scene / video / 'video.mov'
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

        sample = {'image': frame, 'boxes': boxes, 'labels': labels}

        return sample

    
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
            transforms.ToPILImage(),
            transforms.Resize((800, 800)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform

class Trainer:
    def __init__(self, model, optimizer, lr_scheduler, device):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device

    def train_one_epoch(self, data_loader, epoch):
        self.model.train()
        for images, targets in data_loader:
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

# In your main script
if __name__ == "__main__":
    base_path = 'stanford_campus_dataset'
    num_classes = 5  # Including the background class
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Initialize the object detection model
    object_detection_model = ObjectDetectionModel(num_classes)
    model = object_detection_model.model
    model.to(device)

    # Create the dataset with transformations
    dataset_transform = object_detection_model.get_transform()
    dataset = VideoDataset(
        base_path=base_path,
        annotations_path='annotations',
        videos_path='videos',
        transform=dataset_transform
    )

    # Split the dataset into train and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=lambda x: x)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=lambda x: x)

    # Initialize the optimizer and learning rate scheduler
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Initialize the Trainer
    trainer = Trainer(model, optimizer, lr_scheduler, device)

    # Start training
    num_epochs = 10
    trainer.train(train_loader, val_loader, num_epochs)