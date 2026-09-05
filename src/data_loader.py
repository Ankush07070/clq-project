import pandas as pd
import os

def load_metadata(csv_path):
    df = pd.read_csv(csv_path)
    return df

def add_image_paths(df, image_dir):
    image_paths = []
    
    for img_id in df['image_id']:
        path = os.path.join(image_dir, img_id + ".jpg")
        image_paths.append(path)
    
    df['image_path'] = image_paths
    return df

def encode_labels(df):
    label_map = {
        'nv': 0,   # melanocytic nevi
        'mel': 1,  # melanoma
        'bkl': 2,
        'bcc': 3,
        'akiec': 4,
        'vasc': 5,
        'df': 6
    }
    
    df['label'] = df['dx'].map(label_map)
    return df, label_map