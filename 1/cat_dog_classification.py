from fastai.vision.all import *

# fastai’s applications all use the same basic steps and code:
# 1. Create appropriate DataLoaders
# 2. Create a Learner
# 3. Call a fit method
# 4. Make predictions or view results.


def is_cat(x):
    return x[0].isupper()


# Download file model from preset URL of fastai
path = untar_data(URLs.PETS) / "images"
# Create DataLoaders
dls = ImageDataLoaders.from_name_func(
    path,
    get_image_files(path),
    valid_pct=0.2,
    seed=42,
    label_func=is_cat,
    item_tfms=Resize(224),
)

# Create a Learner
learn = vision_learner(dls, resnet34, metrics=error_rate)

# Call a fit method
learn.fine_tune(1)
img = PILImage.create("images/cat.jpg")

# Make predictions and view results
is_cat, _, probs = learn.predict(img)
print(f"Is this a cat?: {is_cat}.")
print(f"Probability it's a cat: {probs[1].item():.6f}")
