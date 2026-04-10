import kagglehub
import shutil
import os

def main():
    print("Downloading dataset via kagglehub...")
    # Download latest version
    path = kagglehub.dataset_download("imsparsh/flowers-dataset")
    print(f"Downloaded to {path}")

    # Set destination relative to project root
    dest = r"c:\Users\ABC\Desktop\anigrevity\Flower_System\data\flowers"
    
    if os.path.exists(dest):
        print(f"Destination {dest} already exists. Cleaning it up first...")
        shutil.rmtree(dest)
        
    print(f"Copying files from {path} to {dest}...")
    shutil.copytree(path, dest)
    print("Dataset copied successfully! Ready for training.")

if __name__ == "__main__":
    main()
