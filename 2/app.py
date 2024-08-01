import gradio as gr
import fastai.vision.all as fast_vision
import pathlib

posix_backup = pathlib.PosixPath

try:
    pathlib.PosixPath = pathlib.WindowsPath
    # Load a Laerner from file, put on GPU
    LEARN = fast_vision.load_learner("hair-resnet18-model.pkl")

finally:
    pathlib.PosixPath = posix_backup

MODELS_PATH = pathlib.Path("./models")
EXAMPLES_PATH = pathlib.Path("./examples")


LABELS = LEARN.dls.vocab


def predict_hair(img):
    # img = PILImage.create(img)
    _, _, probs = LEARN.predict(img)
    return {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}


demo = gr.Interface(
    fn=predict_hair,
    inputs=gr.Image(),
    outputs=gr.Label(num_top_classes=3),
    title="Hair Type Classifier",
    description="A hair type classifier to predict hair type: straight, wavy, curly, kinky, and dreadlocks. Although dreadlocks are not a hair type, we can still classify.",
    examples=[f"{EXAMPLES_PATH}/{f.name}" for f in EXAMPLES_PATH.iterdir()],
)
demo.launch()
