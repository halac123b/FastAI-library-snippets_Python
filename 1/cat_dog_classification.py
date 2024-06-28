import fastai.vision.all as vision

# fastai’s applications all use the same basic steps and code:
# 1. Create appropriate DataLoaders
# 2. Create a Learner
# 3. Call a fit method
# 4. Make predictions or view results.


def is_cat(x):
    return x[0].isupper()


# Download file model from preset URL of fastai
## untar_data(): download file from URL
## URLs: class chứa rất nhiều data và model có sẵn của FastAI
path = vision.untar_data(vision.URLs.PETS) / "images"

# Create DataLoaders: Basic wrapper around several DataLoaders with factory methods for computer vision problems
## Factory method: 1 design pattern - đóng gói sự phức tạp khi phải quản lí cùng lúc nhiều Dataloader (mỗi instance phụ trách 1 dataset và task khác nhau)
## from_name_func(): tạo Dataloader từ path đến dataset và hàm label
dls = vision.ImageDataLoaders.from_name_func(
    path,  # path đến dataset
    vision.get_image_files(path),  # Get tất cả image từ path dataset
    valid_pct=0.2,  # Sử dụng 20% dataset dùng cho bước validation
    seed=42,  # Seed cho các lệnh random
    label_func=is_cat,
    item_tfms=vision.Resize(224),
)

# Create a Learner
learn = vision.vision_learner(dls, vision.resnet34, metrics=vision.error_rate)

# Call a fit method
learn.fine_tune(1)
img = vision.PILImage.create("images/cat.jpg")

# Make predictions and view results
is_cat, _, probs = learn.predict(img)
print(f"Is this a cat?: {is_cat}.")
print(f"Probability it's a cat: {probs[1].item():.6f}")
