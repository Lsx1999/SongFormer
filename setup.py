from setuptools import setup

setup(
    name="songformer",
    version="0.1.0",
    description="SongFormer: Music Generation with AI",
    author="SongFormer Team",
    python_requires=">=3.8",
    packages=[
        "musicfm",
        "musicfm.model",
        "musicfm.modules",
        "SongFormer",
        "SongFormer.dataset",
        "SongFormer.models",
        "SongFormer.postprocessing",
    ],
    package_dir={
        "": "src",
        "musicfm": "src/third_party/musicfm",
        "SongFormer": "src/SongFormer",
    },
)