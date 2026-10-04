import os
import urllib.request
import zipfile

def download_data():
    os.makedirs('data/images', exist_ok=True)
    os.makedirs('data/labels', exist_ok=True)
    
    images_zip = 'data/images.zip'
    if not os.path.exists(images_zip):
        print("Downloading images.zip from Zenodo...")
        urllib.request.urlretrieve("https://zenodo.org/records/7939060/files/images.zip?download=1", images_zip)
        print("Download complete.")
        
    print("Extracting images...")
    with zipfile.ZipFile(images_zip, 'r') as zip_ref:
        # Extract all directly into data/images/
        zip_ref.extractall('data/images/')
    print("Extraction complete.")
    
    base_url = "https://raw.githubusercontent.com/AlexOlsen/DeepWeeds/master/labels"
    for name in ["labels", "train_subset0", "val_subset0", "test_subset0"]:
        csv_path = f"data/labels/{name}.csv"
        if not os.path.exists(csv_path):
            print(f"Downloading {name}.csv...")
            urllib.request.urlretrieve(f"{base_url}/{name}.csv", csv_path)
    print("All labels downloaded.")

if __name__ == "__main__":
    download_data()
