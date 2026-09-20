from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolo11n.pt")

    model.train(
        data=r"add path to your data.yaml file of your data set",
        epochs=300, # will see images 100 time improving each time
        patience=30,  
        imgsz=640,  #resize image to 640*640 to maintain consistency
        batch=16,    #will process 16 images at once to increase efficiency
        device=0 #to use gpu
    )