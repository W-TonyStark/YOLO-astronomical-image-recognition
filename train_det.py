# from ultralytics import YOLO
#
# if __name__ == '__main__':
#     model = YOLO(r'D:\Sheng\sheng\stars\test\ultralytics-main\ultralytics\cfg\models\v8\yolov8-spdconv.yaml')
#     model.load('yolov8n.pt')  # loading pretrain weights
#     model.train(data=r'D:\Sheng\sheng\stars\test\ultralytics-main\stars.yaml', epochs=400, imgsz=950, device=0, batch=2,
#                 workers=0, multi_scale=True)

from ultralytics import YOLO

# import os
# os.environ['CUDA_VISIBLE_DEVICES']='1'
if __name__ == '__main__':
    # Load a model
    model = YOLO(
        r'D:\Sheng\sheng\stars\test\ultralytics-main\ultralytics\cfg\models\v8\yolov8-spdconv.yaml')  # 不使用预训练权重训练
    model.load("yolov8n.pt")  # 使用预训练权重训练
    # Trainparameters ----------------------------------------------------------------------------------------------
    # model.train(
    #     data=r'D:\Sheng\sheng\stars\test\ultralytics-main\stars.yaml',
    #     epochs=400,
    #     batch=2,
    #     imgsz=1000,
    #     workers=0,
    #     device=0,
    #     multi_scale=True
    # )

    model.predict(
        model=r'D:\Sheng\sheng\stars\test\ultralytics-main\runs\detect\train5\weights\best.pt',
        source=r'D:\Sheng\sheng\stars\test\yolov8\tests\images\stars.jpeg',
        save_txt=True,
        conf=0.01,
        max_det=1000,
        iou=0.1,
    )




